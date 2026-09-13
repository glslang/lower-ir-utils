//! Host-side conversions for the chrono constant-lowering wrappers.

#![cfg(feature = "chrono")]

use lower_ir_utils::{JitNaiveDate, JitNaiveDateTime, JitNaiveTime};

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
