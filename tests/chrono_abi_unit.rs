//! Unit tests for the [`JitParam`] mapping of the `chrono` wrappers in
//! `lower_ir_utils::external::chrono`. Mirrors the shape of
//! `tests/abi_unit.rs` — no JIT required, we just call `push_params` and
//! inspect the resulting `AbiParam` vector.

#![cfg(feature = "chrono")]

use cranelift_codegen::ir::{AbiParam, Type, types};
use lower_ir_utils::{JitNaiveDate, JitNaiveDateTime, JitNaiveTime, JitParam};

const PTR64: Type = types::I64;

fn collect<T: JitParam>() -> Vec<AbiParam> {
    let mut v = Vec::new();
    T::push_params(&mut v, PTR64);
    v
}

fn collect_with<T: JitParam>(ptr_ty: Type) -> Vec<AbiParam> {
    let mut v = Vec::new();
    T::push_params(&mut v, ptr_ty);
    v
}

fn types_of(params: &[AbiParam]) -> Vec<Type> {
    params.iter().map(|p| p.value_type).collect()
}

#[test]
fn naive_date_is_one_i32() {
    assert_eq!(types_of(&collect::<JitNaiveDate>()), vec![types::I32]);
}

#[test]
fn naive_time_is_two_i32() {
    assert_eq!(
        types_of(&collect::<JitNaiveTime>()),
        vec![types::I32, types::I32]
    );
}

#[test]
fn naive_date_time_is_three_i32() {
    assert_eq!(
        types_of(&collect::<JitNaiveDateTime>()),
        vec![types::I32, types::I32, types::I32]
    );
}

#[test]
fn composes_in_tuples() {
    // Validates that the wrapper plays nicely with the existing tuple
    // `JitParam` impls in `src/abi.rs` — the tuple expands each element in
    // order, so this should be the three datetime scalars followed by the
    // trailing scalar.
    assert_eq!(
        types_of(&collect::<(JitNaiveDateTime, i64)>()),
        vec![types::I32, types::I32, types::I32, types::I64]
    );
}

#[test]
fn encoding_is_independent_of_ptr_ty() {
    // The wrappers ignore `ptr_ty` — their scalars are always I32.
    assert_eq!(
        types_of(&collect_with::<JitNaiveDate>(types::I32)),
        vec![types::I32],
    );
    assert_eq!(
        types_of(&collect_with::<JitNaiveTime>(types::I32)),
        vec![types::I32, types::I32],
    );
    assert_eq!(
        types_of(&collect_with::<JitNaiveDateTime>(types::I32)),
        vec![types::I32, types::I32, types::I32],
    );
}

#[test]
fn from_conversions_round_trip() {
    use chrono::{NaiveDate, NaiveTime};

    let d = NaiveDate::from_ymd_opt(2026, 5, 21).unwrap();
    let w: JitNaiveDate = d.into();
    let d2: NaiveDate = w.into();
    assert_eq!(d, d2);

    let t = NaiveTime::from_hms_nano_opt(12, 34, 56, 789).unwrap();
    let w: JitNaiveTime = t.into();
    let t2: NaiveTime = w.into();
    assert_eq!(t, t2);

    let dt = d.and_time(t);
    let w: JitNaiveDateTime = dt.into();
    let dt2: chrono::NaiveDateTime = w.into();
    assert_eq!(dt, dt2);
}
