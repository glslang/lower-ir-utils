# ABI and calling conventions

`lower-ir-utils` builds Cranelift signatures and lowers call arguments. Native
signatures support scalars, thin pointers/references, and unit. They reject
by-value tuples, fat pointers (`&str`, `&[T]`, `&mut [T]`), and chrono wrappers.
See [the report assessment](openvuln-assessment.md) for the ABI defects behind
this breaking correction.

## Native signatures and IR arguments

[`JitParam`](../src/abi.rs) describes a native parameter or return type. A
custom implementation requires `unsafe impl`: its scalar encoding must match
Rust's native C argument and return ABI on all supported targets and at every
parameter position. A transparent newtype can delegate to its scalar field.
Matching a type's size or the number of IR values does not establish this
contract. Rust distinguishes memory layout from function-call ABI, and tuples
use an unspecified Rust layout. [Rust Reference](https://doc.rust-lang.org/reference/type-layout.html)

`JitArg` independently lowers constants or existing SSA values into arguments.
For example, a static string still lowers into a data pointer and length, but
the host signature must declare those as **two separate parameters**:

```rust
use lower_ir_utils::jit_export;

#[jit_export]
fn text_len(_data: *const u8, len: usize) -> usize {
    len
}
```

Use `text_len_jit::call(bcx, module, id, "hello".as_ptr(), 5usize)`, or
`jit_call!(bcx, ptr_ty, local; "hello")` with a declared local function reference.
The latter expands one Rust expression into two scalar IR arguments.
`#[jit_export]`'s generated `call` takes one expression per declared parameter.

The same rule applies to chrono constants: `JitNaiveDate` lowers to days from
CE, `JitNaiveTime` to seconds and nanoseconds, and `JitNaiveDateTime` to all
three. Declare separate `i32` parameters and reconstruct using chrono's checked
`*_opt` constructors. Their in-memory chrono values do not use that encoding.

## Thin pointers and ownership

`*const T`, `*mut T`, `&T`, and `&mut T` are supported when `T: Sized`.
A reference to `[T; N]` is a thin pointer; a reference to `[T]` is a fat pointer.
An opaque pointer can refer to a Rust type such as `HashMap` if only Rust
interprets its storage. If JIT code reads fields, the host and JIT must agree on
actual offsets and validity; a suitable representation such as `repr(C)` can
supply the memory-layout contract. It does not automatically implement an
aggregate calling convention.

Embedded addresses must remain valid for every JIT invocation. Static-reference
`JitArg` implementations retain their `'static` bounds. Raw pointers require
caller-managed lifetimes. Mutable access must respect Rust's aliasing rules;
strings additionally require valid UTF-8, and `bool` values must be 0 or 1.
These helpers do not validate arbitrary generated IR or sandbox untrusted code.

For multiple results, use caller-owned output storage:

```rust
use lower_ir_utils::jit_export;

#[jit_export]
fn divmod(a: i64, b: i64, out: &mut [i64; 2]) {
    *out = [a / b, a % b];
}
```

For strings, an output buffer plus capacity and a scalar returned length makes
ownership explicit. Another option is a thin pointer to a host-owned object
with host access/free helpers. The owner must keep storage alive until the
caller finishes using it; the crate supplies no allocator or ownership runtime.

## Calling convention and aggregates

Both [`jit_signature!`](../src/macros.rs) and
[`#[jit_export]`](../macros/src/lib.rs) obtain the convention and pointer width
from `Module::make_signature` and `Module::target_config`. Exported symbols
must be used with a module targeting the host's native C ABI. The attribute
adds `extern "C"` when omitted and rejects explicitly different conventions.

Aggregate ABIs can pack several fields into one register, assign different
register classes, move the whole argument to the stack, or use an indirect
argument/result pointer. On AArch64, for example, large composites and indirect
results have dedicated handling, and the number of available argument registers
matters. [AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)

Expanding fields into independent scalar `AbiParam`s loses the aggregate
information required for those decisions. The earlier claim that SysV/AAPCS
fat pointers always matched two scalar arguments was incorrect, including at
register exhaustion. These signatures are now rejected on every platform.

`IntoReturns` and `define_function` still accept multiple SSA results with a
manually constructed signature for JIT-to-JIT calls. Both generated sides must
use that same convention; these IR results are not Rust tuple returns. The
Cranelift verifier checks internal IR consistency, not agreement with a Rust
function-pointer cast.
