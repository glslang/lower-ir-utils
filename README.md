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
  `signature`, `declare`, and `call` helpers, and auto-injects `extern "C"`
  if no ABI is specified.

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

## Layout

- `src/abi.rs` — `JitParam` / `JitArg` traits and impls.
- `src/builder.rs` — `define_function` + `IntoReturns`.
- `src/macros.rs` — `jit_signature!`, `jit_call!`, `define_jit_fn!`.
- `macros/` — proc-macro crate exporting `#[jit_export]`.
- `tests/` — integration tests (`jit_integration`, `define_function`,
  `abi_unit`, `jit_export`) plus an `external_consumer` workspace.

## Building

```
cargo build
cargo test
```

Targets Cranelift 0.131. Tested on x86_64 Linux (System V ABI). The `&str` /
`&[T]` impls assume the platform passes fat pointers as separate `(ptr, len)`
args; on platforms where that doesn't hold, prefer flat scalar params.

## License

MIT — see `LICENSE`.
