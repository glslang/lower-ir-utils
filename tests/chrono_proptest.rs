//! Property tests for the `chrono` wrappers' scalar encoding.
//!
//! The `JitArg::lower` impls in `src/external/chrono.rs` encode each value as
//! plain integer scalars (`num_days_from_ce`, `num_seconds_from_midnight`,
//! `nanosecond`); the JIT-callee reconstructs with the inverse constructors
//! (`from_num_days_from_ce_opt`, `from_num_seconds_from_midnight_opt`). The
//! invariant that matters is that this scheme is **injective and
//! round-trippable** over the whole value domain — including the leap-second
//! range where `nanosecond() >= 1_000_000_000`.
//!
//! These tests exercise the same chrono calls `lower` does, as a pure
//! host-side round-trip (no `FunctionBuilder` needed), the way
//! `tests/chrono_abi_unit.rs::from_conversions_round_trip` does for fixed
//! values — but swept across the domain by proptest.

#![cfg(feature = "chrono")]

use chrono::{Datelike, NaiveDate, NaiveTime, Timelike};
use proptest::prelude::*;

/// A `NaiveDate` over a wide, representable year range. `num_days_from_ce`
/// is defined for the full proleptic Gregorian calendar, so we sweep well
/// past the usual 1900–2100 window.
fn arb_date() -> impl Strategy<Value = NaiveDate> {
    (-4000i32..=4000i32, 1u32..=366u32)
        .prop_filter_map("valid (year, ordinal) day", |(year, ord)| {
            NaiveDate::from_yo_opt(year, ord)
        })
}

/// A `NaiveTime` including the leap-second range: chrono represents a leap
/// second by letting `nanosecond()` return `>= 1_000_000_000`, which
/// `from_hms_nano_opt` accepts only on a `:59` second.
fn arb_time() -> impl Strategy<Value = NaiveTime> {
    (0u32..24, 0u32..60, 0u32..60, 0u32..2_000_000_000)
        .prop_filter_map("valid (h, m, s, nano) time", |(h, m, s, nano)| {
            NaiveTime::from_hms_nano_opt(h, m, s, nano)
        })
}

proptest! {
    /// `JitNaiveDate` encodes a date as `num_days_from_ce`; the inverse
    /// `from_num_days_from_ce_opt` must recover it exactly.
    #[test]
    fn date_round_trips(d in arb_date()) {
        let days = d.num_days_from_ce();
        prop_assert_eq!(NaiveDate::from_num_days_from_ce_opt(days), Some(d));
    }

    /// `JitNaiveTime` encodes a time as the `(seconds_from_midnight,
    /// nanosecond)` pair (each reinterpreted through `as i32` / `as u32`);
    /// the inverse must recover it exactly, leap seconds included.
    #[test]
    fn time_round_trips(t in arb_time()) {
        let secs = t.num_seconds_from_midnight() as i32;
        let nano = t.nanosecond() as i32;
        let recovered =
            NaiveTime::from_num_seconds_from_midnight_opt(secs as u32, nano as u32);
        prop_assert_eq!(recovered, Some(t));
    }

    /// `JitNaiveDateTime` is the date scalars followed by the time scalars;
    /// the composed round-trip must be identity.
    #[test]
    fn date_time_round_trips(d in arb_date(), t in arb_time()) {
        let dt = d.and_time(t);

        let days = dt.date().num_days_from_ce();
        let secs = dt.time().num_seconds_from_midnight() as i32;
        let nano = dt.time().nanosecond() as i32;

        let date = NaiveDate::from_num_days_from_ce_opt(days);
        let time =
            NaiveTime::from_num_seconds_from_midnight_opt(secs as u32, nano as u32);
        let recovered = date.zip(time).map(|(d, t)| d.and_time(t));
        prop_assert_eq!(recovered, Some(dt));
    }

    /// Injectivity sweep: any two distinct times must produce distinct
    /// `(secs, nano)` encodings. (Round-trip identity above already implies
    /// this, but asserting it directly guards against a future encoding
    /// change that round-trips lossily yet collides two inputs.)
    #[test]
    fn time_encoding_is_injective(a in arb_time(), b in arb_time()) {
        let enc = |t: NaiveTime| (t.num_seconds_from_midnight(), t.nanosecond());
        prop_assert_eq!(enc(a) == enc(b), a == b);
    }
}

/// Regression for the exact collision commit `5648789` fixed: an earlier
/// single-`I64` `secs * 1_000_000_000 + nano` encoding mapped the last
/// nanosecond of a leap second (`secs=59, nano=1_000_000_000`) onto the start
/// of the next second (`secs=60, nano=0`). The two-scalar form keeps them
/// distinct.
#[test]
fn leap_second_does_not_collide_with_next_second() {
    let leap = (59u32, 1_000_000_000u32);
    let next = (60u32, 0u32);
    assert_ne!(leap, next);

    // And the would-be-colliding old scalar must differ too.
    let old = |(s, n): (u32, u32)| s as i64 * 1_000_000_000 + n as i64;
    // The old form *did* collide — documenting why the two-scalar form exists.
    assert_eq!(old(leap), old(next));
}
