# Repository Guidelines

## Project Structure & Module Organization

`lower-ir-utils` is a Rust 2024 workspace for Cranelift JIT helper APIs, currently pinned to Cranelift `0.132` and Rust `1.93`. The root crate lives in `src/`: `abi.rs` defines `JitParam` and `JitArg`, `builder.rs` contains function-definition helpers, `macros.rs` exports declarative macros, and `lib.rs` handles re-exports plus the hidden `__reexport` module used by macros. Feature-gated modules are `disasm.rs` (`disas`), `sim.rs` (`sim`), `runtime.rs` (`tokio`), and `external/chrono.rs` (`chrono`). The `macros/` member is the proc-macro crate for `#[jit_export]`. Integration tests are in `tests/`; `tests/external_consumer/` is excluded from the workspace and checks downstream use without direct Cranelift dependencies.

## Build, Test, and Development Commands

- `cargo build` builds the root crate and the `macros` workspace member.
- `cargo test` runs default-feature integration and doc tests with the standard Cargo runner.
- `cargo test --features disas,sim,chrono,tokio` exercises all optional feature modules locally.
- `cargo nextest run --workspace --all-targets` matches the main CI test runner.
- `cargo test -p lower-ir-utils-macros` tests the proc-macro crate only.
- `cargo test --manifest-path tests/external_consumer/Cargo.toml` or `cd tests/external_consumer && cargo nextest run --all-targets` verifies macro hygiene from a downstream crate.
- `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass before PRs.
- `cargo doc --workspace --no-deps --all-features` checks rustdoc generation, matching docs.rs and CI.
- `cargo llvm-cov nextest --workspace --all-targets --lcov --output-path lcov.info` matches the coverage workflow when coverage changes matter.

## Coding Style & Naming Conventions

Use rustfmt and keep CI warning-free; CI sets `RUSTFLAGS=-D warnings` and docs set `RUSTDOCFLAGS=-D warnings -D rustdoc::broken_intra_doc_links`. Public APIs should include concise rustdoc matching the existing style, with `ignore` examples when snippets need a live Cranelift `Module`. Declarative macros should reference dependencies through `$crate::__reexport::...` instead of assuming downstream crates imported Cranelift or `smallvec`. Keep default dependencies limited; optional integrations should stay behind Cargo features.

## Testing Guidelines

Add integration coverage in `tests/` for public behavior and API changes. Feature-specific tests are gated with `#![cfg(feature = "...")]`: `disasm.rs`, `sim.rs` and `sim_proptest.rs`, `chrono_*`, and `tokio_runtime.rs`. Keep `JitParam::push_params` and `JitArg::lower` consistent: if a type contributes N ABI params, lowering must emit N values in the same order. For proc-macro changes, test both `tests/jit_export.rs` and the external consumer crate. Miri is scoped to `cargo miri test --test abi_unit` because JIT execution cannot run under Miri.

## Commit & Pull Request Guidelines

Recent history uses short imperative or scoped commit subjects, for example `docs: refresh CLAUDE.md and README for 0.132 + feature modules` and `chore: migrate to Rust edition 2024`. Keep commits focused and explain API or CI changes in the body when needed. PRs should describe the change, list tests run, link related issues, and call out platform or ABI implications, especially for fat pointers, Windows MSVC behavior, async/JIT boundary changes, feature gates, or Cranelift version changes.

## Agent-Specific Notes

Prefer reading the small source files directly before changing behavior. Do not broaden `'static` assumptions for `&str`, slices, or reference/pointer `JitArg` impls without addressing JIT lifetime safety; embedded host addresses must outlive every JIT invocation. `#[jit_export]` intentionally rejects `async fn`; bridge async work through a synchronous shim instead of changing the generated ABI. Avoid hiding Cranelift verifier failures with retries or extra unwraps, and keep cross-platform behavior in mind because CI runs Linux, macOS, and Windows across x86_64/aarch64 where available.
