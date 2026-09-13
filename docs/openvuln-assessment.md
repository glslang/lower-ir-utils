# OpenVuln report assessment

Reviewed `openvuln-glslang-lower-ir-utils-full.md` against this repository on
2026-09-13. The 15 findings describe related defects in one native ABI design:
`JitParam` flattened Rust aggregates into independent scalar `AbiParam`s,
while rustc compiled the corresponding functions using aggregate ABI rules.
The underlying defects apply to the pre-fix code. The report's exploitation
claims and severity scores are not independently established by this review.

## Evidence and qualifications

The report audits Cranelift 0.134.3; this checkout actually resolves 0.135.1
with chrono 0.4.45. Both signature macros still used the same unrestricted
mapping, the proc macro still suppressed `improper_ctypes_definitions`, and
chrono's wrappers still encoded days/seconds/nanoseconds independently of their
memory representation. Inspection of Cranelift 0.135.1's x64 and AArch64
`compute_arg_locs` confirms that separate scalar parameters are allocated
individually. Aggregate grouping is not recovered from those parameters.

A harmless wrong-value probe against an isolated copy of the pre-fix `HEAD`
reproduced the date mismatch locally (Rust 1.98.0, aarch64-apple-darwin,
Cranelift 0.135.1). An identity JIT function declared with
`fn(JitNaiveDate) -> i32`, returning its first IR parameter, received
**16,599,258** for 2026-05-21 instead of the documented days-from-CE encoding
**739,757**. The input was a valid chrono date; the returned value was an `i32`.
No invalid reference, arbitrary memory access, or exploit chain was needed.

Rust does not guarantee tuple field order or a two-word layout for pointers
to unsized types. Memory layout alone also does not establish function-call
compatibility. [Rust Reference](https://doc.rust-lang.org/reference/type-layout.html)
AArch64's aggregate classification, register allocation, and indirect-result
rules likewise differ from independent scalar parameters.
[AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)

The earlier claim that Linux/macOS/AArch64 were unaffected was incorrect.
However, this review does not establish that every signature or every call
corrupts values: some shapes happen to agree on some targets. The report itself
refutes its original two-field tuple-return reorder examples. It also mentions
a one-element tuple workaround for a chrono return, although the original
`JitParam` tuple implementations only covered arities 2 through 6.

Arbitrary read/write or code execution requires a consumer that exposes the
relevant signatures and attacker influence over values or programs. This
library is not a standalone network service or an untrusted-code sandbox.
The report's modeled services, ASLR leaks, and execution chains were not rerun;
their exact behavior on other toolchains/targets remains unverified here.
These qualifications do not make the confirmed ABI mismatch acceptable.

## Finding disposition

“Applies” below confirms the unsafe mapping mechanism in the pre-fix code,
not every payload, address, register value, or severity assertion in the report.
The `BUG-` keys identify the original findings.

| Finding | Assessment | Fix |
| --- | --- | --- |
| `BUG-R2-S1-A4-H1` | Applies: chrono date memory bits are not days from CE; locally reproduced. | Remove `JitParam` for `JitNaiveDate`; retain scalar constant lowering. |
| `BUG-R2-S1-A4-H2` | Applies: chrono time/datetime are aggregates, not two/three independent `i32` parameters. | Remove their `JitParam` implementations. |
| `BUG-R2-S1-A4-H3` | Applies: chrono return encoding/packing also differs; the generated scalar call helper additionally exposed only its first result. | Reject all chrono wrapper return types. |
| `BUG-R2-S2-A1-H1` | Applies: tuple declaration order is not a native ABI layout contract. | Reject tuple parameters. |
| `BUG-R2-S2-A1-H2` | Applies: scalar flattening omits SysV MEMORY-class aggregate argument handling. | Reject tuple parameters of every size. |
| `BUG-R2-S2-A1-H3` | Applies: packed small fields consume different registers and can shift following arguments. | Reject packed tuple parameters. |
| `BUG-R2-S2-A2-H1` | Applies: large host tuple returns may require a hidden result pointer absent from the signature. | Reject tuple returns from exports. |
| `BUG-R2-S2-A2-H2` | Applies: JIT tuple returns have the same missing native result-buffer protocol in the other direction. | Reject tuple returns in `jit_signature!` / `define_jit_fn!`. |
| `BUG-R2-S2-A2-H3` | Applies to layout/classification mismatches; original two-field reorder examples are not supported by the report's own later analysis. | Reject all tuple returns, including nested/reference-bearing shapes. |
| `BUG-R2-S2-A2-H4` | Applies: small tuple returns can pack into fewer native registers than the IR results. | Reject small tuple returns. |
| `BUG-R2-S2-A3-H1` | Applies: separate fat-pointer lanes lose SysV aggregate register-exhaustion handling. | Reject `&str`, `&[T]`, and `&mut [T]` in signatures at every position. |
| `BUG-R2-S2-A4-H1` | Applies: flattened returns do not describe AArch64's indirect-result protocol. | Reject tuple returns on AArch64 too. |
| `BUG-R2-S2-A4-H2` | Applies: scalar fat-pointer lanes lose AArch64 aggregate allocation at register exhaustion. | Reject fat-pointer signatures on AArch64 too. |
| `BUG-R2-S2-A4-H3` | Applies: large AArch64 composite parameters can be indirect instead of independent scalar values. | Reject large tuple parameters on AArch64 too. |
| `BUG-R2-S2-A4-H4` | Applies: mixed-class or packed AArch64 tuples need aggregate classification, not per-field scalar classes. | Reject mixed-class and packed tuple signatures. |

## Implementation and compatibility

The fix removes aggregate `JitParam` implementations rather than maintaining
partial target-specific allowlists. Requiring eight-byte tuple fields does
not address large aggregates, mixed register classes, or register exhaustion.
A `size_of <= 16` check does not handle packed fields or indirect Microsoft
x64 aggregates. Syntactic name checks miss aliases. The shared trait boundary
rejects all these cases, including aliases, on all targets.

`JitParam` is now an unsafe trait with an explicit scalar/unit native ABI
contract, so downstream custom implementations must justify their encoding.
The proc macro no longer suppresses the FFI lint or offers tuple-return helper
behavior. It also rejects explicitly non-C calling conventions, since its
signature generator always uses the module's native default convention.

This is a **breaking API correction** and should be released accordingly.
The crate versions have not been changed or published. Migration is to separate
scalar parameters and explicit caller-owned output storage. Static string/slice
and chrono `JitArg` lowering remains available, with existing lifetime bounds.
Multi-result IR remains supported through explicit Cranelift signatures for
JIT-to-JIT calls. See [the ABI guide](abi-and-calling-conventions.md) and
[README](../README.md#native-abi-and-migration) for examples.

The fix does not validate arbitrary SSA values, raw pointers, manual
signatures, function-pointer casts, or unsafe custom implementations. Callers
must still satisfy Rust validity, lifetime, and aliasing requirements.

## Regression coverage

- Independent compile-fail doctests cover tuple packing, reordering, size,
  nesting, references, fat pointers, and all three chrono wrappers. They cover
  exports and both signature-building macros, aliases, and unsafe custom impls.
- Native host → JIT → host tests exercise separate pointer/length parameters
  after 5, 7, and 9 integer arguments, with a trailing reference and float.
- Output-pointer and JIT-to-JIT multiple-result tests preserve supported use
  cases without pretending that SSA results are a native Rust tuple.
- Chrono tests preserve nanoseconds and leap seconds; a checked scalar shim
  handles day zero, rejects invalid fields, and writes through a trailing output
  pointer only after validation.
- The external consumer checks scalar calls and compile-time rejection without
  direct Cranelift dependencies.

Local execution is on aarch64 macOS. The existing CI matrix must still validate
Linux and Windows on x86_64/aarch64; those platforms were not executed locally.

Local validation completed successfully:

- `cargo test --workspace --offline` (default features).
- `cargo nextest run --workspace --all-targets --all-features --offline`
  (65 passed, none skipped).
- `cargo test --workspace --doc --all-features --offline`
  (30 compile-fail tests passed; 12 existing example doctests ignored).
- `cargo test --manifest-path tests/external_consumer/Cargo.toml --offline`
  (3 runtime tests and 3 compile-fail doctests passed).
- `cargo fmt --all -- --check` and
  `cargo clippy --workspace --all-targets --all-features --offline -- -D warnings`.
- `cargo doc --workspace --no-deps --all-features --offline` with
  `RUSTDOCFLAGS='-D warnings -D rustdoc::broken_intra_doc_links'`.
