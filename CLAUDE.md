# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`lower-ir-utils` is a thin layer over Cranelift's JIT API (all `cranelift-*`
crates pinned to `0.134`). It is **not** a compiler — it provides plumbing to:

1. Convert Rust types into Cranelift `AbiParam`s / `Signature`s (`JitParam`).
2. Lower Rust values (constants or already-lowered `Value`s) into IR `Value`s
   at call sites (`JitArg`).
3. Reduce boilerplate around declaring + defining functions in a `Module`.

Every public item has thorough rustdoc; prefer reading the source over
guessing. `README.md` is the canonical user-facing reference and carries
runnable examples for each feature.

## Workspace layout

- Root crate `lower-ir-utils` (`src/`):
  - `abi.rs` — `JitParam` / `JitArg` traits, plus impls for scalars,
    pointers, references, `&str`, `&[T]`, and small tuples.
  - `builder.rs` — `define_function` + `IntoReturns`.
  - `macros.rs` — `jit_signature!`, `jit_call!`, `define_jit_fn!`.
  - `lib.rs` — re-exports and a hidden `__reexport` module that the macros
    use to reference Cranelift / smallvec without forcing dependents to add
    them to their own `Cargo.toml`.
  - `disasm.rs` — `define_function_with_disasm`, `format_disassembly`
    (feature `disas`, pulls in `capstone`).
  - `sim.rs` — `Simulator` / `SimValue` / `SimResult`, a small Cranelift IR
    interpreter over a flat byte buffer (feature `sim`, no extra deps).
  - `external/` — foreign-type `JitParam`/`JitArg` wrappers; `chrono.rs`
    today (feature `chrono`).
  - `runtime.rs` — `spawn_blocking_build` async helper (feature `tokio`).
- Workspace member `macros/` — proc-macro crate exporting `#[jit_export]`.
- `tests/external_consumer/` — **excluded** from the workspace; verifies the
  crate works (and that macro hygiene holds) as an external dependency.

## Cargo features

All optional and **off by default**; each gates one module:

- `disas` → `disasm` module (side-by-side machine-code / mnemonic dumps via
  Capstone).
- `sim` → `sim` module (IR interpreter; host `call`s are stubbed with
  zero-valued results — a debug aid, not a runtime).
- `chrono` → `external::chrono` (`JitNaiveDate` / `JitNaiveTime` /
  `JitNaiveDateTime`, lowering to scalar immediates — no host pointers).
- `tokio` → `runtime` module (`spawn_blocking_build`; depends on tokio with
  only the `rt` feature).

Tests for these modules only compile/run with the matching feature enabled.

## Building & testing

```
cargo build                                  # root + macros crates
cargo test                                   # default-feature tests
cargo test --features disas,sim,chrono,tokio # exercise the optional modules
cargo test -p lower-ir-utils-macros          # macros crate alone
cargo test --test jit_integration            # a single integration test file
cargo nextest run --workspace --all-targets --all-features  # what CI actually runs
```

No feature is on by default, so `--all-features` is what makes the gated
suites (`sim*`, `disasm`, `chrono_*`, `tokio_runtime`) compile at all — drop
it and they become empty test binaries that report success without running.

`cranelift-native` is a dev-dependency, so tests JIT-compile and **execute**
generated code against the host ISA; they won't run on a target without a
Cranelift backend.

CI (`.github/workflows/`) sets `RUSTFLAGS=-D warnings` and gates on the
following — run them before opening a PR:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --no-deps --all-features   # RUSTDOCFLAGS=-D warnings
```

CI runs the test matrix on x86_64 + aarch64 Linux, aarch64 macOS, and x86_64
+ aarch64 Windows, so changes must hold across ABIs (see the `&str`/`&[T]`
fat-pointer note below). Doctests and the `external_consumer` crate run as
separate steps. Miri is scoped to `cargo miri test --test abi_unit` — the
only test with no JIT invocation, since Cranelift's JIT path (FFI + generated
machine code) can't execute under Miri.

## Conventions

- **Macros must reference Cranelift through `$crate::__reexport::...`**, not
  by assuming the consumer has the dep in scope. Same for `smallvec`.
- **`JitParam` and `JitArg` must stay self-consistent.** If a type pushes N
  `AbiParam`s in `push_params`, its `JitArg::lower` must emit exactly N
  `Value`s in the same order. Mismatches surface as confusing Cranelift
  verifier errors at runtime.
- **`#[jit_export]` auto-injects `extern "C"`** when no ABI is given and
  silences `improper_ctypes_definitions` so `&str` etc. are usable. Don't
  remove that without a plan for the lints it'll re-enable. It also **rejects
  `async fn` with a compile error** — JIT IR is synchronous machine code, and
  an `async extern "C" fn` would compile into a silent ABI mismatch (real
  return is an opaque future). Bridge async work through a sync shim.
- **Doc comments are first-class.** Match the existing thorough rustdoc style
  for new public items. Use `# Example` blocks marked ```ignore``` since most
  snippets need a live `Module`.
- **No `unsafe` outside of obvious FFI test glue** (e.g. `mem::transmute` of
  `get_finalized_function` results in tests).

## When changing the API

- Update `tests/` to cover the new shape — there's no separate example dir.
- If you touch the macro crate, check `tests/external_consumer/` still
  compiles (`cargo test --manifest-path tests/external_consumer/Cargo.toml`);
  the macros' hygiene is easy to break by accident.
- Keep `lib.rs` re-exports tight: only items consumers need at the top level.
- New optional functionality should usually land behind a Cargo feature
  rather than expanding the default dependency set.

## Things to avoid

- Don't pull in extra dependencies casually; the default dep list is
  intentionally short (Cranelift + smallvec + the proc-macro toolchain).
  Optional deps (`capstone`, `chrono`, `tokio`) must stay feature-gated.
- Don't paper over Cranelift verifier failures with `unwrap` retries — they
  almost always indicate a `JitParam` / `JitArg` mismatch.
- Don't broaden `&'static` bounds on `JitArg for &str` / `&[T]` without
  thinking through lifetime: the data pointer is embedded as an immediate in
  the IR, so it must outlive every JIT invocation.
