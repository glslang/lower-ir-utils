//! End-to-end JIT round-trips for the `chrono` wrappers in
//! `lower_ir_utils::external::chrono`. Each test passes a wrapped chrono
//! value as a single argument expression to a callee, relying on the
//! `JitArg` impl to lower it to the matching IR scalars and on the callee
//! signature (declared via `JitParam`) to receive them. The host body just
//! combines the scalars so we can assert exact round-trip values.
//!
//! [`JitNaiveDate`] is one Rust expression → one IR value, so the simple
//! `#[jit_export]`-generated `call()` works. [`JitNaiveTime`] and
//! [`JitNaiveDateTime`] are one Rust expression → multiple IR values, so
//! those tests use [`jit_call!`] directly — its argument-expression list is
//! independent of the callee's Rust parameter list.
//!
//! The block-creation, return, and finalize boilerplate lives in
//! `define_function` — same setup as `tests/jit_integration.rs`.

#![cfg(feature = "chrono")]

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use lower_ir_utils::{
    define_jit_fn, jit_call, jit_export, JitNaiveDate, JitNaiveDateTime, JitNaiveTime,
};

fn jit_builder() -> JITBuilder {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    JITBuilder::with_isa(isa, default_libcall_names())
}

// ------------------------------------------------------------------
// JitNaiveDate -> single I32 scalar.
// ------------------------------------------------------------------

#[jit_export]
fn echo_days(d: i32) -> i32 {
    d
}

#[test]
fn lowers_naive_date_to_i32_days_from_ce() {
    let mut jb = jit_builder();
    echo_days_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = echo_days_jit::declare(&mut module);

    let date = NaiveDate::from_ymd_opt(2026, 5, 21).unwrap();
    let expected_days = date.num_days_from_ce();

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn() -> i32,
        |bcx, module, _params| { echo_days_jit::call(bcx, module, ext_id, JitNaiveDate(date)) },
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i32 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    let got = f();
    assert_eq!(got, expected_days);
    assert_eq!(NaiveDate::from_num_days_from_ce_opt(got).unwrap(), date);
}

// ------------------------------------------------------------------
// JitNaiveTime -> (I32 secs_from_midnight, I32 nanosecond). One Rust
// expression lowers into two IR values, so we use `jit_call!` directly
// against a 2-arg callee.
// ------------------------------------------------------------------

#[jit_export]
fn combine_time(secs: i32, nano: i32) -> i64 {
    // Reconstruct the (otherwise non-injective) nanos-from-midnight encoding
    // so the test asserts both scalars survived the trip.
    (secs as u32 as i64) * 1_000_000_000 + (nano as u32 as i64)
}

fn assert_time_round_trips(time: NaiveTime) {
    let mut jb = jit_builder();
    combine_time_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = combine_time_jit::declare(&mut module);

    let expected =
        time.num_seconds_from_midnight() as i64 * 1_000_000_000 + time.nanosecond() as i64;

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn() -> i64,
        |bcx, module, _params| {
            let ptr_ty = module.target_config().pointer_type();
            let local = module.declare_func_in_func(ext_id, bcx.func);
            let inst = jit_call!(bcx, ptr_ty, local; JitNaiveTime(time));
            bcx.inst_results(inst)[0]
        },
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(), expected);
    assert_eq!(
        NaiveTime::from_num_seconds_from_midnight_opt(
            time.num_seconds_from_midnight(),
            time.nanosecond(),
        )
        .unwrap(),
        time,
    );
}

#[test]
fn lowers_naive_time_to_two_scalars() {
    assert_time_round_trips(NaiveTime::from_hms_nano_opt(12, 34, 56, 789_000_001).unwrap());
}

#[test]
fn naive_time_leap_second_is_distinct_from_next_second() {
    // (secs=59, nano=1_000_000_000) — the start-of-minute leap-second
    // representation — and (secs=60, nano=0) collapsed to the same i64
    // under the previous `secs * 1e9 + nano` encoding. With the two-scalar
    // form they're distinct; round-trip both end-to-end to prove it.
    let leap = NaiveTime::from_num_seconds_from_midnight_opt(59, 1_000_000_000).unwrap();
    let next = NaiveTime::from_num_seconds_from_midnight_opt(60, 0).unwrap();
    assert_ne!(leap, next);
    assert_time_round_trips(leap);
    assert_time_round_trips(next);
}

// ------------------------------------------------------------------
// JitNaiveDateTime -> (I32, I32, I32). One Rust expression lowers into
// three IR values; use `jit_call!` again.
// ------------------------------------------------------------------

#[jit_export]
fn combine_dt(days: i32, secs: i32, nano: i32) -> i64 {
    // Mixes all three scalars so a wire-up regression on any of them fails
    // the test. The constants are arbitrary but distinct.
    (days as i64)
        .wrapping_mul(1_000_000_007)
        .wrapping_add((secs as u32 as i64).wrapping_mul(1_000_003))
        .wrapping_add(nano as u32 as i64)
}

#[test]
fn lowers_naive_date_time_to_three_scalars() {
    let mut jb = jit_builder();
    combine_dt_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = combine_dt_jit::declare(&mut module);

    let date = NaiveDate::from_ymd_opt(2026, 5, 21).unwrap();
    let time = NaiveTime::from_hms_nano_opt(12, 34, 56, 789).unwrap();
    let dt = NaiveDateTime::new(date, time);
    let days = date.num_days_from_ce();
    let secs = time.num_seconds_from_midnight();
    let nano = time.nanosecond();
    let expected = (days as i64)
        .wrapping_mul(1_000_000_007)
        .wrapping_add((secs as i64).wrapping_mul(1_000_003))
        .wrapping_add(nano as i64);

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn() -> i64,
        |bcx, module, _params| {
            let ptr_ty = module.target_config().pointer_type();
            let local = module.declare_func_in_func(ext_id, bcx.func);
            // Single argument expression -> 3 IR values via JitArg.
            let inst = jit_call!(bcx, ptr_ty, local; JitNaiveDateTime(dt));
            bcx.inst_results(inst)[0]
        },
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(), expected);
}
