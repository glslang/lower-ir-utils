//! Traits describing how Rust types map onto Cranelift's ABI.
//!
//! [`JitParam`] is type-level: it answers "what `AbiParam`s does this Rust type
//! contribute to a Cranelift `Signature`?". [`JitArg`] is value-level: it answers
//! "how do I lower this Rust value into one or more Cranelift `Value`s at a call site?".
//!
//! The two traits are intentionally independent — at a call site some arguments
//! come from already-lowered IR `Value`s (function parameters, prior instructions)
//! while others are constants embedded into the IR. The macros do not enforce a
//! correspondence between the signature types and the argument expressions; that
//! is the caller's responsibility, exactly as it would be for `bcx.ins().call`.
//!
//! # Native signatures
//!
//! Signatures support scalars, thin pointers/references, and unit. Tuples,
//! `&str`, slices, and chrono wrappers are not native signature types: expanding
//! an aggregate into scalar lanes does not implement its platform C ABI.
//! Pass separate scalar parameters and use explicit output pointers for multiple
//! results. [`JitArg`] can still expand strings, slices, and chrono constants
//! for calls to functions that explicitly declare those scalar parameters.

use cranelift_codegen::ir::{AbiParam, InstBuilder, Type, Value, types};
use cranelift_frontend::FunctionBuilder;
use smallvec::SmallVec;

/// Type-level mapping for a Rust type with a scalar (or unit) native C ABI.
///
/// Tuples, fat pointers, and chrono wrappers deliberately do not implement this
/// trait, including through type aliases. Use separate scalar parameters or
/// an explicit pointer to caller-owned output storage instead.
///
/// # Safety
///
/// An implementation must append exactly one scalar parameter, or none for a
/// type ABI-equivalent to unit. Its encoding must match Rust's `extern "C"`
/// argument AND return ABI on every supported target, at every parameter
/// position (including register exhaustion). Matching size or field order alone
/// is insufficient. If the type also implements [`JitArg`], its lowered values
/// must match that encoding, count, and order.
///
/// Prefer delegating to the scalar field for a `#[repr(transparent)]` newtype.
/// Do not implement this trait by flattening an aggregate.
#[doc = include_str!("../docs/abi-rejections.md")]
pub unsafe trait JitParam {
    /// Append this type's parameter to `out`, or nothing for unit.
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type);
}

/// Value-level mapping: a Rust value lowers into zero or more Cranelift `Value`s.
///
/// Implementations are provided for already-lowered `Value`s (passthrough),
/// integer/float constants (emitted as `iconst`/`f64const`), `&'static str`
/// and `&'static [T]` (emitted as ptr+len constants), `&'static T` /
/// `&'static mut T` (emitted as a single `iconst`), and raw pointers (emitted
/// as `iconst`).
///
/// # Pointer / reference lifetimes
///
/// Anything that lowers into a pointer immediate (`*const T`, `*mut T`,
/// `&'static T`, `&'static mut T`, `&'static str`, `&'static [T]`) embeds the
/// host address into the IR as a constant. The pointee must therefore outlive
/// **every invocation** of the JIT-compiled function — a stack or
/// short-lived heap pointer captured here is a use-after-free waiting to
/// happen. The `&'static`-bounded impls enforce this at the type level; the
/// raw-pointer impls leave it to the caller (see their `# Safety` notes).
pub trait JitArg {
    /// Lower `self` into one or more cranelift values, appended to `out`.
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>);
}

// ---------- JitParam impls ----------

// SAFETY: Unit contributes no argument or return value.
unsafe impl JitParam for () {
    fn push_params(_out: &mut Vec<AbiParam>, _ptr_ty: Type) {}
}

macro_rules! impl_jit_param_scalar {
    ($($t:ty => $cl:expr),* $(,)?) => {
        $(
            // SAFETY: This primitive has the corresponding scalar C ABI.
            unsafe impl JitParam for $t {
                fn push_params(out: &mut Vec<AbiParam>, _ptr_ty: Type) {
                    out.push(AbiParam::new($cl));
                }
            }
        )*
    };
}

impl_jit_param_scalar! {
    i8  => types::I8,
    u8  => types::I8,
    i16 => types::I16,
    u16 => types::I16,
    i32 => types::I32,
    u32 => types::I32,
    i64 => types::I64,
    u64 => types::I64,
    f32 => types::F32,
    f64 => types::F64,
    bool => types::I8,
}

macro_rules! impl_jit_param_pointerlike {
    ($($t:ty),* $(,)?) => {
        $(
            // SAFETY: This primitive has the corresponding scalar C ABI.
            unsafe impl JitParam for $t {
                fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
                    out.push(AbiParam::new(ptr_ty));
                }
            }
        )*
    };
}

impl_jit_param_pointerlike!(usize, isize);

// SAFETY: Sized pointees make this a thin pointer in the native C ABI.
unsafe impl<T: Sized> JitParam for *const T {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        out.push(AbiParam::new(ptr_ty));
    }
}

// SAFETY: Sized pointees make this a thin pointer in the native C ABI.
unsafe impl<T: Sized> JitParam for *mut T {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        out.push(AbiParam::new(ptr_ty));
    }
}

// SAFETY: Sized pointees make this a thin pointer in the native C ABI.
unsafe impl<T: Sized> JitParam for &T {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        out.push(AbiParam::new(ptr_ty));
    }
}

// SAFETY: Sized pointees make this a thin pointer in the native C ABI.
unsafe impl<T: Sized> JitParam for &mut T {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        out.push(AbiParam::new(ptr_ty));
    }
}

// ---------- JitArg impls ----------

impl JitArg for Value {
    fn lower(self, _bcx: &mut FunctionBuilder, _ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(self);
    }
}

impl<const N: usize> JitArg for [Value; N] {
    fn lower(self, _bcx: &mut FunctionBuilder, _ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.extend(self);
    }
}

macro_rules! impl_jit_arg_iconst {
    ($($t:ty => $cl:expr),* $(,)?) => {
        $(
            impl JitArg for $t {
                fn lower(self, bcx: &mut FunctionBuilder, _ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
                    out.push(bcx.ins().iconst($cl, self as i64));
                }
            }
        )*
    };
}

impl_jit_arg_iconst! {
    i8  => types::I8,
    u8  => types::I8,
    i16 => types::I16,
    u16 => types::I16,
    i32 => types::I32,
    u32 => types::I32,
    i64 => types::I64,
    u64 => types::I64,
    bool => types::I8,
}

impl JitArg for usize {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self as i64));
    }
}

impl JitArg for isize {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self as i64));
    }
}

impl JitArg for f32 {
    fn lower(self, bcx: &mut FunctionBuilder, _ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().f32const(self));
    }
}

impl JitArg for f64 {
    fn lower(self, bcx: &mut FunctionBuilder, _ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().f64const(self));
    }
}

/// Embeds the raw pointer as an `iconst` immediate.
///
/// # Safety
///
/// `JitArg` itself is a safe trait, but using this impl requires the same
/// discipline you would apply when handing a raw pointer across an FFI
/// boundary: the address baked into the IR must remain valid for the entire
/// lifetime of every JIT invocation that may execute the resulting code, and
/// any reads/writes performed through it on the JIT side must respect Rust's
/// aliasing and validity rules for the pointee type. Stack pointers,
/// heap allocations that may be freed before the IR is dropped, and reborrowed
/// references to short-lived data are footguns. Prefer
/// [`JitArg for &'static T`](#impl-JitArg-for-%26'static+T) when you can.
impl<T: Sized> JitArg for *const T {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self as i64));
    }
}

/// Embeds the raw pointer as an `iconst` immediate.
///
/// # Safety
///
/// Same caveats as the `*const T` impl above, plus: if the JIT side issues
/// writes through this pointer, they must not alias any other live access to
/// the pointee on the host side. Prefer
/// [`JitArg for &'static mut T`](#impl-JitArg-for-%26'static+mut+T) where
/// applicable.
impl<T: Sized> JitArg for *mut T {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self as i64));
    }
}

/// Embeds the reference's address as an `iconst` immediate.
///
/// The `'static` bound guarantees the pointee outlives any JIT invocation;
/// this is the recommended path for embedding host data into generated code.
impl<T: Sized> JitArg for &'static T {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self as *const T as i64));
    }
}

/// Embeds the reference's address as an `iconst` immediate.
///
/// The `'static` bound guarantees the pointee outlives any JIT invocation. As
/// with any `&'static mut`, the caller is the sole live mutable borrow at the
/// moment the immediate is captured; preventing further aliasing across JIT
/// invocations is the caller's responsibility.
impl<T: Sized> JitArg for &'static mut T {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self as *mut T as i64));
    }
}

/// Embeds the string's data pointer and length as IR constants.
///
/// The `'static` bound guarantees the bytes outlive every JIT invocation.
impl JitArg for &'static str {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self.as_ptr() as i64));
        out.push(bcx.ins().iconst(ptr_ty, self.len() as i64));
    }
}

/// Embeds the slice's data pointer and length as IR constants.
///
/// The `'static` bound guarantees the elements outlive every JIT invocation.
impl<T: Sized> JitArg for &'static [T] {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        out.push(bcx.ins().iconst(ptr_ty, self.as_ptr() as i64));
        out.push(bcx.ins().iconst(ptr_ty, self.len() as i64));
    }
}
