//! End-to-end JIT round-trips for the `chrono` wrappers in
//! `lower_ir_utils::external::chrono`. Each test passes a wrapped chrono
//! value as a single argument expression to a `#[jit_export]`-generated
//! `call`, relying on the `JitArg` impl to lower it to the matching IR
//! scalars and on the `#[jit_export]` host signature (declared via
//! `JitParam`) to receive them. The host body just echoes / combines the
//! scalars so we can assert exact round-trip values.
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
// JitNaiveTime -> single I64 scalar.
// ------------------------------------------------------------------

#[jit_export]
fn echo_nanos(n: i64) -> i64 {
    n
}

#[test]
fn lowers_naive_time_to_i64_nanos_from_midnight() {
    let mut jb = jit_builder();
    echo_nanos_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = echo_nanos_jit::declare(&mut module);

    let time = NaiveTime::from_hms_nano_opt(12, 34, 56, 789_000_001).unwrap();
    let expected_nanos =
        time.num_seconds_from_midnight() as i64 * 1_000_000_000 + time.nanosecond() as i64;

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn() -> i64,
        |bcx, module, _params| { echo_nanos_jit::call(bcx, module, ext_id, JitNaiveTime(time)) },
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    let got = f();
    assert_eq!(got, expected_nanos);

    let secs = (got / 1_000_000_000) as u32;
    let nanos = (got % 1_000_000_000) as u32;
    assert_eq!(
        NaiveTime::from_num_seconds_from_midnight_opt(secs, nanos).unwrap(),
        time,
    );
}

// ------------------------------------------------------------------
// JitNaiveDateTime -> (I32, I64). One wrapper-arg lowers into two IR values
// the same way `&'static str` lowers into (ptr, len). The `#[jit_export]`
// `call()` shape requires one Rust expression per Rust parameter, so to
// pass a *single* `JitNaiveDateTime` expression we go through `jit_call!`
// directly — it accepts any number of arg expressions and runs each through
// `JitArg::lower`.
// ------------------------------------------------------------------

#[jit_export]
fn combine_dt(days: i32, nanos: i64) -> i64 {
    // Mixes both scalars so a wire-up regression on either side fails the test.
    (days as i64)
        .wrapping_mul(1_000_000_003)
        .wrapping_add(nanos)
}

#[test]
fn lowers_naive_date_time_to_two_scalars() {
    let mut jb = jit_builder();
    combine_dt_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = combine_dt_jit::declare(&mut module);

    let date = NaiveDate::from_ymd_opt(2026, 5, 21).unwrap();
    let time = NaiveTime::from_hms_nano_opt(12, 34, 56, 789).unwrap();
    let dt = NaiveDateTime::new(date, time);
    let days = date.num_days_from_ce();
    let nanos = time.num_seconds_from_midnight() as i64 * 1_000_000_000 + time.nanosecond() as i64;
    let expected = (days as i64)
        .wrapping_mul(1_000_000_003)
        .wrapping_add(nanos);

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn() -> i64,
        |bcx, module, _params| {
            let ptr_ty = module.target_config().pointer_type();
            let local = module.declare_func_in_func(ext_id, bcx.func);
            // Single argument expression -> 2 IR values via JitArg.
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
