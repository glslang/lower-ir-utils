# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

`lower-ir-utils` is a thin layer over Cranelift's JIT API (crate version
`0.131`). It is **not** a compiler — it provides plumbing to:

1. Convert Rust types into Cranelift `AbiParam`s / `Signature`s.
2. Lower Rust values (constants or already-lowered `Value`s) into IR `Value`s
   at call sites.
3. Reduce boilerplate around declaring + defining functions in a `Module`.

The crate is small (a few hundred lines). Prefer reading the source over
guessing — every public item has a doc comment.

## Workspace layout

- Root crate `lower-ir-utils` (`src/`):
  - `abi.rs` — `JitParam` and `JitArg` traits, plus impls for scalars,
    pointers, references, `&str`, `&[T]`, and small tuples.
  - `builder.rs` — `define_function` + `IntoReturns`.
  - `macros.rs` — `jit_signature!`, `jit_call!`, `define_jit_fn!`
    declarative macros.
  - `lib.rs` — re-exports and a hidden `__reexport` module that the macros
    use to reference Cranelift / smallvec without forcing dependents to add
    them to their own `Cargo.toml`.
- Workspace member `macros/` — proc-macro crate exporting `#[jit_export]`.
- `tests/external_consumer/` — excluded from the workspace; verifies the
  crate works as an external dependency.

## Conventions

- **Macros must reference Cranelift through `$crate::__reexport::...`**, not
  by assuming the consumer has the dep in scope. Same for `smallvec`.
- **`JitParam` and `JitArg` must stay self-consistent.** If you add a new
  type that pushes N `AbiParam`s in `push_params`, its `JitArg::lower` must
  emit exactly N `Value`s in the same order. Mismatches surface as confusing
  Cranelift verifier errors at runtime.
- **`#[jit_export]` auto-injects `extern "C"`** when no ABI is given and
  silences `improper_ctypes_definitions` so `&str` etc. are usable. Don't
  remove that without a plan for the lints it'll re-enable.
- **Doc comments are first-class.** Existing code has thorough rustdoc;
  match that style for new public items. Use `# Example` blocks marked
  ```ignore``` since most snippets need a live `Module`.
- **No `unsafe` outside of obvious FFI test glue** (e.g. `mem::transmute` of
  `get_finalized_function` results in tests).

## Building & testing

```
cargo build            # builds root + macros crates
cargo test             # runs tests/ — JIT-compiles and executes generated code
cargo test -p lower-ir-utils-macros   # macros crate alone
```

`cranelift-native` is a dev-dependency, so tests pick up the host ISA; they
will not run on a target without a Cranelift backend.

## When changing the API

- Update `tests/` to cover the new shape — there's no separate example
  directory.
- If you touch the macro crate, check `tests/external_consumer/` still
  compiles; the macros' hygiene is easy to break by accident.
- Keep `lib.rs` re-exports tight: only items consumers need at the top
  level.

## Things to avoid

- Don't pull in extra dependencies casually; the dep list is intentionally
  short (Cranelift + smallvec + the proc-macro toolchain).
- Don't paper over Cranelift verifier failures with `unwrap` retries —
  they almost always indicate a `JitParam` / `JitArg` mismatch.
- Don't broaden `&'static` bounds on `JitArg for &str` / `&[T]` without
  thinking through lifetime: the data pointer is embedded as an immediate
  in the IR, so it must outlive every JIT invocation.
