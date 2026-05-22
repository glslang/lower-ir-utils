# lower-ir-utils

[![CI](https://github.com/glslang/lower-ir-utils/actions/workflows/ci.yml/badge.svg)](https://github.com/glslang/lower-ir-utils/actions/workflows/ci.yml)
[![Miri](https://github.com/glslang/lower-ir-utils/actions/workflows/miri.yml/badge.svg)](https://github.com/glslang/lower-ir-utils/actions/workflows/miri.yml)
[![codecov](https://codecov.io/gh/glslang/lower-ir-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/glslang/lower-ir-utils)
[![crates.io](https://img.shields.io/crates/v/lower-ir-utils.svg)](https://crates.io/crates/lower-ir-utils)
[![docs.rs](https://img.shields.io/docsrs/lower-ir-utils)](https://docs.rs/lower-ir-utils)
[![License: MIT](https://img.shields.io/crates/l/lower-ir-utils.svg)](LICENSE)

Helpers for bridging Rust types to [Cranelift](https://cranelift.dev/) JIT
signatures and call sites. The crate trims the boilerplate of declaring
external functions, building `Signature`s, and lowering arguments, while still
giving you direct access to the underlying `FunctionBuilder` and `Module`.

## What it provides

- **`JitParam`** — type-level: how a Rust type expands into Cranelift
  `AbiParam`s (e.g. `&str` becomes `(ptr, len)`).
- **`JitArg`** — value-level: how a Rust value lowers into one or more
  `cranelift_codegen::ir::Value`s. Implemented for already-lowered `Value`s,
  integer/float constants, raw pointers, and `&'static str` / `&'static [T]`.
- **`jit_signature!(&module; fn(T1, T2) -> R)`** — build a `Signature` from
  Rust types.
- **`jit_call!(&mut bcx, ptr_ty, callee; arg1, arg2, ...)`** — emit a `call`
  instruction, lowering each argument through `JitArg`.
- **`define_function(...)` / `define_jit_fn!(...)`** — declare + define a
  function in one shot. Your closure receives the `FunctionBuilder`, the
  `Module`, and the entry-block params; whatever it returns is funneled
  through `IntoReturns` and emitted as `return_`.
- **`#[jit_export]`** (proc-macro) — annotate a Rust function so it can be
  called from JITed IR. Generates a sibling `<fn>_jit` module with `register`,
  `signature`, `declare`, `try_declare`, and `call` helpers, and auto-injects
  `extern "C"` if no ABI is specified.
- **`<fn>_jit::try_declare`** — fallible variant of `declare`, returning
  `ModuleResult<FuncId>` so signature conflicts surface as errors instead
  of panics.
- **`JitArg for &'static T` / `&'static mut T`** — embed a host reference
  as a single pointer immediate. Preferred over raw pointers when you
  already have a real Rust reference; the `'static` bound guarantees the
  pointee outlives every JIT invocation.
- **`disasm` module** (feature `disas`) — `define_function_with_disasm`
  and `format_disassembly` produce a side-by-side opcode / mnemonic dump
  of the compiled bytes via Capstone.
- **`sim` module** (feature `sim`) — `Simulator` / `SimResult` walk a
  finalized `Function` against a flat byte buffer, with optional
  per-instruction trace. A debug aid, not a runtime — host `call`s are
  stubbed.
- **`JitNaiveDate` / `JitNaiveTime` / `JitNaiveDateTime`** (feature
  `chrono`) — `JitParam`/`JitArg` newtypes for `chrono` naive date/time,
  lowering to scalar immediates.

## Example

```rust
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use lower_ir_utils::{define_jit_fn, jit_export};

#[jit_export]
fn double_i64(x: i64) -> i64 {
    x.wrapping_mul(2)
}

fn build(module: &mut JITModule) {
    let ext_id = double_i64_jit::declare(module);

    let _wrap_id = define_jit_fn!(
        module, "wrap", Linkage::Export, fn(i64) -> i64,
        |bcx, module, params| {
            double_i64_jit::call(bcx, module, ext_id, params[0])
        },
    ).unwrap();
}
```

See `tests/jit_integration.rs` and `tests/define_function.rs` for end-to-end
examples covering multi-value returns, `&str` arguments, and slice arguments.

### Tuple returns

For a `#[jit_export]` function whose Rust signature is a tuple,
`<fn>_jit::call` returns `cranelift_codegen::ir::Inst`, not an array of
`Value`s. Pull the results out via `bcx.inst_results(inst)`:

```rust
#[jit_export]
fn divmod(a: i64, b: i64) -> (i64, i64) {
    (a / b, a % b)
}

let inst = divmod_jit::call(&mut bcx, &mut module, ext_id, a, b);
let results: Vec<_> = bcx.inst_results(inst).to_vec();
bcx.ins().return_(&results);
```

The `Inst` shape is intentional: tuple elements with multi-lane `JitParam`
impls (`&str`, `&[T]`, nested tuples) push more than one `AbiParam` each,
so the ABI return count and the syntactic tuple arity can diverge — a
fixed-arity `[Value; N]` shape would silently drop lanes. See
`tests/jit_export.rs` for a `(&'static str, i64)` regression case.

## Cargo features

All optional features are off by default. Enable them with
`cargo add lower-ir-utils --features disas,sim,chrono` (or
`cargo test --features disas,sim,chrono`).

### `disas` — machine-code disassembly

Pulls in `capstone` (the same disassembler Cranelift uses behind its own
`disas` feature) and exposes `define_function_with_disasm` plus the
lower-level `format_disassembly`. Works transparently on x86_64,
aarch64, riscv64, and s390x.

```rust
use cranelift_module::Linkage;
use lower_ir_utils::{define_function_with_disasm, jit_signature};

let (id, dump) = define_function_with_disasm(
    &mut module,
    "wrap",
    Linkage::Export,
    jit_signature!(&module; fn(i64) -> i64),
    |_bcx, _module, params| params[0],
)?;
println!("{}", dump.text);
// 0x0000  48 89 f8           mov rax, rdi
// 0x0003  c3                 ret
```

### `sim` — Cranelift IR interpreter

A small interpreter that walks a finalized `Function` block by block
against a flat `Vec<u8>` memory, then renders the trace, SSA register
table, memory hexdump, and return values. Covers scalar int/float
arithmetic, bitwise / shift ops, `icmp` / `fcmp`, `select`, `load` /
`store`, branching, and casts. Host `call` instructions are stubbed
(zero-valued results) — useful for sanity-checking the IR *around* a
call without leaving the simulator.

```rust
use lower_ir_utils::sim::{SimValue, Simulator};

let mut sim = Simulator::with_memory(vec![0; 256]);
sim.trace = true;
let result = sim.run(&func, &[SimValue::I64(10), SimValue::I64(32)]);
result.dump();
```

### `chrono` — naive date/time wrappers

Pulls in `chrono` (default features off) and exposes newtype wrappers in
`lower_ir_utils::external::chrono` (re-exported at the crate root as
`JitNaiveDate`, `JitNaiveTime`, `JitNaiveDateTime`). Each lowers to plain
`iconst` scalars — no host pointers — so non-`'static` values are fine.
`JitNaiveTime` uses two `I32` lanes (seconds from midnight, then
nanoseconds) so leap-second encodings round-trip injectively.

```rust
use chrono::NaiveDate;
use lower_ir_utils::{jit_export, JitNaiveDate};

#[jit_export]
fn days_since_ce(d: i32) -> i32 {
    d
}

// …declare `days_since_ce` in your module, then:
let date = NaiveDate::from_ymd_opt(2026, 5, 21).unwrap();
days_since_ce_jit::call(&mut bcx, &mut module, ext_id, JitNaiveDate(date));
```

See `tests/chrono_jit_integration.rs` for end-to-end JIT coverage.

## Layout

- `src/abi.rs` — `JitParam` / `JitArg` traits and impls.
- `src/builder.rs` — `define_function` + `IntoReturns`.
- `src/macros.rs` — `jit_signature!`, `jit_call!`, `define_jit_fn!`.
- `src/disasm.rs` — `define_function_with_disasm`, `format_disassembly`
  (feature `disas`).
- `src/sim.rs` — `Simulator`, `SimValue`, `SimResult` (feature `sim`).
- `src/external/` — foreign-type `JitParam`/`JitArg` wrappers (feature-gated;
  `chrono` submodule today).
- `macros/` — proc-macro crate exporting `#[jit_export]`.
- `tests/` — integration tests (`jit_integration`, `define_function`,
  `abi_unit`, `jit_export`, `disasm`, `sim`, `chrono_*`) plus an
  `external_consumer` workspace.

## Building

```
cargo build
cargo test
cargo test --features disas,sim,chrono    # exercises the optional modules
```

Targets Cranelift 0.131. Tested on x86_64 Linux (System V ABI). The `&str` /
`&[T]` impls assume the platform passes fat pointers as separate `(ptr, len)`
args; on platforms where that doesn't hold, prefer flat scalar params.

## License

MIT — see `LICENSE`.
