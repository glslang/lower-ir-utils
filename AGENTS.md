# Repository Guidelines

## Project Structure & Module Organization

`lower-ir-utils` is a Rust 2021 workspace for Cranelift JIT helper APIs. The root crate lives in `src/`: `abi.rs` defines `JitParam` and `JitArg`, `builder.rs` contains function-definition helpers, `macros.rs` exports declarative macros, and `lib.rs` handles re-exports. The `macros/` member is the proc-macro crate for `#[jit_export]`. Integration tests are in `tests/`; `tests/external_consumer/` is excluded from the workspace and checks downstream use.

## Build, Test, and Development Commands

- `cargo build` builds the root crate and the `macros` workspace member.
- `cargo test` runs the local integration and doc tests with the standard Cargo runner.
- `cargo nextest run --workspace --all-targets` matches the main CI test runner.
- `cargo test -p lower-ir-utils-macros` tests the proc-macro crate only.
- `cargo test --manifest-path tests/external_consumer/Cargo.toml` verifies macro hygiene from a downstream crate.
- `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass before PRs.
- `cargo doc --workspace --no-deps` checks rustdoc generation.

## Coding Style & Naming Conventions

Use rustfmt and keep CI warning-free; CI sets `RUSTFLAGS=-D warnings`. Public APIs should include concise rustdoc matching the existing style, with `ignore` examples when snippets need a live Cranelift `Module`. Declarative macros should reference dependencies through `$crate::__reexport::...` instead of assuming downstream crates imported Cranelift or `smallvec`. Keep new dependencies rare and justified.

## Testing Guidelines

Add integration coverage in `tests/` for public behavior and API changes. Keep `JitParam::push_params` and `JitArg::lower` consistent: if a type contributes N ABI params, lowering must emit N values in the same order. For proc-macro changes, test both `tests/jit_export.rs` and the external consumer crate. Miri is scoped to `cargo miri test --test abi_unit` because JIT execution cannot run under Miri.

## Commit & Pull Request Guidelines

Recent history uses short imperative or scoped commit subjects, for example `ci: set explicit GITHUB_TOKEN permissions for CodeQL` and `Improve rustdoc for docs.rs, tighten CI docs job`. Keep commits focused and explain API or CI changes in the body when needed. PRs should describe the change, list tests run, link related issues, and call out platform or ABI implications, especially for fat pointers, Windows MSVC behavior, or Cranelift version changes.

## Agent-Specific Notes

Prefer reading the small source files directly before changing behavior. Do not broaden `'static` assumptions for `&str` or slices without addressing JIT lifetime safety, and avoid hiding Cranelift verifier failures with retries or extra unwraps.
