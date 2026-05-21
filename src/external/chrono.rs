//! [`JitParam`] / [`JitArg`] wrappers for `chrono`'s naive date/time types.
//!
//! Enabled by the `chrono` Cargo feature. Each wrapper is a `pub` newtype
//! around the corresponding `chrono` type and lowers to plain integer
//! constants:
//!
//! | Wrapper             | ABI shape          | Encoding                                                                  |
//! |---------------------|--------------------|---------------------------------------------------------------------------|
//! | [`JitNaiveDate`]    | one `I32`          | [`NaiveDate::num_days_from_ce`]                                           |
//! | [`JitNaiveTime`]    | one `I64`          | `num_seconds_from_midnight() * 1_000_000_000 + nanosecond()`              |
//! | [`JitNaiveDateTime`]| `I32` then `I64`   | the two above, in that order (date first, time second)                    |
//!
//! Reconstruction on the JIT-callee side uses the inverse constructors:
//! [`NaiveDate::from_num_days_from_ce_opt`],
//! [`NaiveTime::from_num_seconds_from_midnight_opt`], and pairing the two
//! for [`NaiveDateTime`].
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
//! return a value `>= 1_000_000_000` on the affected second. The
//! [`JitNaiveTime`] encoding preserves that — the host can re-detect the
//! leap second by passing the same nanosecond value back through
//! [`NaiveTime::from_num_seconds_from_midnight_opt`], which accepts it.

use ::chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use cranelift_codegen::ir::{AbiParam, Type, Value};
use cranelift_frontend::FunctionBuilder;
use smallvec::SmallVec;

use crate::abi::{JitArg, JitParam};

/// Newtype wrapper carrying a [`NaiveDate`] across the JIT ABI boundary as a
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct JitNaiveDate(pub NaiveDate);

/// Newtype wrapper carrying a [`NaiveTime`] across the JIT ABI boundary as a
/// single `I64` scalar (nanoseconds from midnight).
///
/// The encoding is
/// `num_seconds_from_midnight() as i64 * 1_000_000_000 + nanosecond() as i64`;
/// leap-second nanos (`>= 1e9`) are passed through unchanged so a host-side
/// call to [`NaiveTime::from_num_seconds_from_midnight_opt`] round-trips
/// the original value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct JitNaiveTime(pub NaiveTime);

/// Newtype wrapper carrying a [`NaiveDateTime`] across the JIT ABI boundary
/// as two scalars: an `I32` (days from CE) followed by an `I64` (nanos from
/// midnight). The ordering mirrors how `&str` lowers to `(ptr, len)`.
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

#[inline]
fn nanos_from_midnight(t: NaiveTime) -> i64 {
    t.num_seconds_from_midnight() as i64 * 1_000_000_000 + t.nanosecond() as i64
}

impl JitParam for JitNaiveDate {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        <i32 as JitParam>::push_params(out, ptr_ty);
    }
}

impl JitArg for JitNaiveDate {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        self.0.num_days_from_ce().lower(bcx, ptr_ty, out);
    }
}

impl JitParam for JitNaiveTime {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        <i64 as JitParam>::push_params(out, ptr_ty);
    }
}

impl JitArg for JitNaiveTime {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        nanos_from_midnight(self.0).lower(bcx, ptr_ty, out);
    }
}

impl JitParam for JitNaiveDateTime {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        <JitNaiveDate as JitParam>::push_params(out, ptr_ty);
        <JitNaiveTime as JitParam>::push_params(out, ptr_ty);
    }
}

impl JitArg for JitNaiveDateTime {
    fn lower(self, bcx: &mut FunctionBuilder, ptr_ty: Type, out: &mut SmallVec<[Value; 8]>) {
        JitNaiveDate(self.0.date()).lower(bcx, ptr_ty, out);
        JitNaiveTime(self.0.time()).lower(bcx, ptr_ty, out);
    }
}
