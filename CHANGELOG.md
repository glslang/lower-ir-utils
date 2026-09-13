# Changelog

## 0.4.0 — 2026-09-13

This release corrects native ABI mismatches that could corrupt values and
pointers when JIT code calls Rust functions or Rust calls JIT functions.
Both `lower-ir-utils` and `lower-ir-utils-macros` are released as 0.4.0.

### Breaking changes

- **Native signatures reject tuples, fat pointers, and chrono wrappers.**
  This applies to `#[jit_export]`, `jit_signature!`, and `define_jit_fn!`,
  including type aliases. The old flattening into independent scalar lanes
  did not implement native aggregate packing, stack allocation, or indirect
  argument/result conventions. The restriction applies on all targets.
- **Custom `JitParam` implementations require `unsafe impl`.** The mapping
  must match the native C argument and return ABI, including its value encoding
  and behavior when argument registers are exhausted.
- **`#[jit_export]` rejects explicitly non-C ABIs** and no longer suppresses
  `improper_ctypes_definitions`.
- **Cranelift moves from 0.134 to 0.135.** Consumers with direct Cranelift
  dependencies must use matching 0.135 versions.
- **The root crate now requires Rust 1.95**, matching Cranelift 0.135's minimum
  version. The proc-macro crate's own minimum remains Rust 1.93.

### Migration

Use separate scalar parameters for tuple fields and separate data-pointer and
`usize` length parameters for strings/slices. Use explicit caller-owned output
storage for multiple results:

```rust
use lower_ir_utils::jit_export;

#[jit_export]
fn divmod(a: i64, b: i64, out: &mut [i64; 2]) {
    *out = [a / b, a % b];
}
```

For chrono values, declare separate `i32` parameters for days from CE, seconds
from midnight, and nanoseconds; reconstruct them with chrono's checked `*_opt`
constructors. `JitNaiveDate`, `JitNaiveTime`, and `JitNaiveDateTime` retain
`JitArg` constant lowering. Static string/slice lowering also remains available.
Use `jit_call!` when one expression supplies several scalar parameters.

`IntoReturns` still supports multiple SSA results for JIT-to-JIT calls using
explicit Cranelift signatures. These results do not describe Rust tuple returns.
Existing pointer lifetime, validity, and aliasing obligations remain in effect.

See the [ABI guide](docs/abi-and-calling-conventions.md) and
[finding-by-finding assessment](docs/openvuln-assessment.md) for details. The
assessment confirms the underlying ABI defects and distinguishes them from
exploit chains that were not independently reproduced.

### Validation

The ABI fix in [PR #71](https://github.com/glslang/lower-ir-utils/pull/71) passed
CI on x86_64/AArch64 Linux, AArch64 macOS, and x86_64/AArch64 Windows, plus
formatting, Clippy, rustdoc, Miri, and coverage. Regression coverage includes
65 tests with all features, 30 compile-fail doctests, and the external consumer's
3 runtime tests and 3 compile-fail doctests.
