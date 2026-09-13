//! [`JitArg`] constant-lowering wrappers for `chrono`'s naive date/time types.
//!
//! Enabled by the `chrono` Cargo feature. Each wrapper is a `pub` newtype
//! around the corresponding `chrono` type and lowers to plain integer
//! constants:
//!
//! | Wrapper              | Scalar lanes           | Encoding                                                                  |
//! |----------------------|---------------------|---------------------------------------------------------------------------|
//! | [`JitNaiveDate`]     | one `I32`           | [`NaiveDate::num_days_from_ce`]                                           |
//! | [`JitNaiveTime`]     | `I32` then `I32`    | `num_seconds_from_midnight()`, then [`Timelike::nanosecond`]              |
//! | [`JitNaiveDateTime`] | `I32`, `I32`, `I32` | the two above concatenated (date scalars first, time scalars second)      |
//!
//! Reconstruction on the JIT-callee side uses the inverse constructors:
//! [`NaiveDate::from_num_days_from_ce_opt`] for the date,
//! [`NaiveTime::from_num_seconds_from_midnight_opt`] for the time (it
//! accepts the leap-second `nano >= 1_000_000_000` case unchanged), and
//! pairing the two for [`NaiveDateTime`].
//!
//! These wrappers are **not** native parameter or return types and do not
//! implement [`crate::JitParam`]. Their chrono memory representation differs
//! from these scalar encodings. Declare separate `i32` parameters, lower a
//! wrapper with [`crate::jit_call!`], and validate the scalars with the `*_opt`
//! constructors in the host shim. For multiple results, use output pointers.
//!
//! # Lifetimes
//!
//! Unlike the pointer-bearing [`JitArg`] impls in [`crate::abi`]
//! (`&'static str`, `&'static T`, `*const T`, …), these wrappers emit only
//! scalar `iconst` immediates — there is no host-memory referent baked into
//! the IR, so they carry no lifetime or aliasing obligations and accept
//! non-`'static` values freely.
//!
//! # Leap seconds
//!
//! `chrono` represents a leap second by letting [`Timelike::nanosecond`]
//! return a value `>= 1_000_000_000` on the affected second. Both the
//! seconds-from-midnight and nanosecond scalars are carried verbatim, so a
//! leap-second [`NaiveTime`] round-trips exactly through
//! [`NaiveTime::from_num_seconds_from_midnight_opt`] without colliding with
//! the next second's encoding — an earlier single-`I64`
//! `secs * 1_000_000_000 + nano` form had that collision (e.g.
//! `(secs=59, nano=1_000_000_000)` matched `(secs=60, nano=0)`).

use ::chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use cranelift_codegen::ir::{Type, Value};
use cranelift_frontend::FunctionBuilder;
use smallvec::SmallVec;

use crate::abi::JitArg;

/// Constant-lowering wrapper encoding a [`NaiveDate`] for explicit scalar JIT calls as a
/// single `I32` scalar (days from year 1 CE).
///
/// The JIT-callee side should reconstruct the date with
/// [`NaiveDate::from_num_days_from_ce_opt`].
///
/// # Example
///
/// ```ignore
/// use chrono::NaiveDate;
/// use lower_ir_utils::{define_jit_fn, jit_export, JitNaiveDate};
///
/// #[jit_export]
/// fn echo_days(d: i32) -> i32 { d }
///
/// // …set up `module` and `ext_id` as in the crate-level example…
/// let wrap_id = define_jit_fn!(
///     &mut module, "wrap", Linkage::Export, fn(i32) -> i32,
///     |bcx, module, _params| {
///         let date = NaiveDate::from_ymd_opt(2026, 5, 21).unwrap();
///         echo_days_jit::call(bcx, module, ext_id, JitNaiveDate(date))
///     },
/// )?;
/// ```
///
/// This wrapper cannot be used as a native parameter or return type:
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDate as Alias;
/// #[lower_ir_utils::jit_export]
/// fn invalid(_: Alias) {}
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDate as Alias;
/// #[lower_ir_utils::jit_export]
/// fn invalid() -> Alias { panic!() }
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDate as Alias;
/// fn build(module: &lower_ir_utils::__reexport::cranelift_jit::JITModule) {
///     let _ = lower_ir_utils::jit_signature!(module; fn(Alias));
/// }
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDate as Alias;
/// fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
///     let _ = lower_ir_utils::define_jit_fn!(
///         module, "invalid", lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
///         fn() -> Alias, |_, _, _| (),
///     );
/// }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct JitNaiveDate(pub NaiveDate);

/// Constant-lowering wrapper encoding a [`NaiveTime`] for explicit scalar JIT calls as
/// two `I32` scalars: `num_seconds_from_midnight()` first, then
/// [`Timelike::nanosecond`].
///
/// Splitting the seconds and nanoseconds into two scalars (rather than
/// folding them into one `I64`) keeps the mapping injective for
/// `chrono`'s leap-second representation, where `nanosecond()` may exceed
/// `1_000_000_000`. The host can reconstruct the value with
/// [`NaiveTime::from_num_seconds_from_midnight_opt(secs, nano)`].
///
/// This wrapper cannot be used as a native parameter or return type:
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveTime as Alias;
/// #[lower_ir_utils::jit_export]
/// fn invalid(_: Alias) {}
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveTime as Alias;
/// #[lower_ir_utils::jit_export]
/// fn invalid() -> Alias { panic!() }
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveTime as Alias;
/// fn build(module: &lower_ir_utils::__reexport::cranelift_jit::JITModule) {
///     let _ = lower_ir_utils::jit_signature!(module; fn(Alias));
/// }
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveTime as Alias;
/// fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
///     let _ = lower_ir_utils::define_jit_fn!(
///         module, "invalid", lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
///         fn() -> Alias, |_, _, _| (),
///     );
/// }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct JitNaiveTime(pub NaiveTime);

/// Constant-lowering wrapper encoding a [`NaiveDateTime`] for explicit scalar JIT calls
/// as three `I32` scalars: days-from-CE, seconds-from-midnight, nanosecond.
/// The order is date scalars first, then time scalars — the same
/// concatenation [`JitNaiveDate`] and [`JitNaiveTime`] would produce on
/// their own.
///
/// This wrapper cannot be used as a native parameter or return type:
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDateTime as Alias;
/// #[lower_ir_utils::jit_export]
/// fn invalid(_: Alias) {}
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDateTime as Alias;
/// #[lower_ir_utils::jit_export]
/// fn invalid() -> Alias { panic!() }
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDateTime as Alias;
/// fn build(module: &lower_ir_utils::__reexport::cranelift_jit::JITModule) {
///     let _ = lower_ir_utils::jit_signature!(module; fn(Alias));
/// }
/// ```
///
/// ```compile_fail,E0277
/// use lower_ir_utils::JitNaiveDateTime as Alias;
/// fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
///     let _ = lower_ir_utils::define_jit_fn!(
///         module, "invalid", lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
///         fn() -> Alias, |_, _, _| (),
///     );
/// }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct JitNaiveDateTime(pub NaiveDateTime);

impl From<NaiveDate> for JitNaiveDate {
    #[inline]
    fn from(d: NaiveDate) -> Self {
        JitNaiveDate(d)
    }
}

impl From<JitNaiveDate> for NaiveDate {
    #[inline]
    fn from(w: JitNaiveDate) -> Self {
        w.0
    }
}

impl From<NaiveTime> for JitNaiveTime {
    #[inline]
    fn from(t: NaiveTime) -> Self {
        JitNaiveTime(t)
    }
}

impl From<JitNaiveTime> for NaiveTime {
    #[inline]
    fn from(w: JitNaiveTime) -> Self {
        w.0
    }
}

impl From<NaiveDateTime> for JitNaiveDateTime {
    #[inline]
    fn from(dt: NaiveDateTime) -> Self {
        JitNaiveDateTime(dt)
    }
}

impl From<JitNaiveDateTime> for NaiveDateTime {
    #[inline]
    fn from(w: JitNaiveDateTime) -> Self {
        w.0
    }
}

impl JitArg for JitNaiveDate {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        self.0.num_days_from_ce().lower(bcx, ptr_ty, out);
    }
}

impl JitArg for JitNaiveTime {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        // `as i32` reinterprets the `u32` bit pattern; the host recovers the
        // original `u32` with `as u32` before calling
        // `NaiveTime::from_num_seconds_from_midnight_opt`.
        (self.0.num_seconds_from_midnight() as i32).lower(bcx, ptr_ty, out);
        (self.0.nanosecond() as i32).lower(bcx, ptr_ty, out);
    }
}

impl JitArg for JitNaiveDateTime {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        JitNaiveDate(self.0.date()).lower(bcx, ptr_ty, out);
        JitNaiveTime(self.0.time()).lower(bcx, ptr_ty, out);
    }
}
