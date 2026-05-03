//! Tests for `define_function` (function form) and `define_jit_fn!` (macro
//! form): `IntoReturns` shapes, multi-return, void return, function vs. macro
//! parity.

use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use lower_ir_utils::{define_function, define_jit_fn, jit_export, jit_signature};

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
// 1. Single-Value return — closure body returns a `Value`, IntoReturns
//    wraps it into `[Value; 1]` for the `return_` instruction.
// ------------------------------------------------------------------

#[test]
fn body_returning_value_works() {
    let mut module = JITModule::new(jit_builder());

    let id = define_jit_fn!(
        &mut module,
        "id",
        Linkage::Export,
        fn(i64) -> i64,
        |bcx, _module, params| bcx.ins().iadd_imm(params[0], 1),
    )
    .unwrap();

    module.finalize_definitions().unwrap();
    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    assert_eq!(f(41), 42);
}

// ------------------------------------------------------------------
// 2. Unit return — closure body returns `()`, IntoReturns yields an empty
//    list, function emits `return_(&[])`.
// ------------------------------------------------------------------

use std::sync::atomic::{AtomicI64, Ordering};

static SINK: AtomicI64 = AtomicI64::new(0);

#[jit_export]
fn record(x: i64) {
    SINK.store(x, Ordering::SeqCst);
}

#[test]
fn body_returning_unit_works() {
    let mut jb = jit_builder();
    record_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = record_jit::declare(&mut module);

    let id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn(i64),
        |bcx, module, params| {
            // call returns Inst (unit ret), discarded; closure returns ().
            let _ = record_jit::call(bcx, module, ext_id, params[0]);
        },
    )
    .unwrap();

    module.finalize_definitions().unwrap();
    let f: extern "C" fn(i64) = unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    SINK.store(0, Ordering::SeqCst);
    f(31415);
    assert_eq!(SINK.load(Ordering::SeqCst), 31415);
}

// ------------------------------------------------------------------
// 3. Multi-return via `[Value; N]`. Cranelift signatures support multiple
//    returns; the `IntoReturns for [Value; N]` impl threads them through.
// ------------------------------------------------------------------

#[test]
fn body_returning_array_works() {
    let mut module = JITModule::new(jit_builder());

    let id = define_jit_fn!(
        &mut module,
        "swap_and_double",
        Linkage::Export,
        fn(i64, i64) -> (i64, i64),
        |bcx, _module, params| {
            // Return (b, a*2).
            let doubled = bcx.ins().imul_imm(params[0], 2);
            [params[1], doubled]
        },
    )
    .unwrap();

    module.finalize_definitions().unwrap();
    // SystemV x86_64: two i64 returns come back in (rax, rdx) which matches
    // a Rust `(i64, i64)` repr-Rust tuple in extern "C".
    let f: extern "C" fn(i64, i64) -> (i64, i64) =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    assert_eq!(f(3, 7), (7, 6));
}

// ------------------------------------------------------------------
// 4. Function form (`define_function`) accepts a hand-built Signature and
//    produces the same result as the macro form. Demonstrates that the
//    macro is a thin wrapper.
// ------------------------------------------------------------------

#[test]
fn function_form_matches_macro_form() {
    let mut module = JITModule::new(jit_builder());

    // Macro form.
    let macro_id = define_jit_fn!(
        &mut module,
        "macro_form",
        Linkage::Export,
        fn(i64) -> i64,
        |bcx, _module, params| bcx.ins().iadd_imm(params[0], 100),
    )
    .unwrap();

    // Function form — same logic, hand-built signature.
    let sig = jit_signature!(&module; fn(i64) -> i64);
    let fn_id = define_function(
        &mut module,
        "fn_form",
        Linkage::Export,
        sig,
        |bcx, _module, params| bcx.ins().iadd_imm(params[0], 100),
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let m: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(macro_id)) };
    let g: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(fn_id)) };
    assert_eq!(m(7), 107);
    assert_eq!(g(7), 107);
}

// ------------------------------------------------------------------
// 5. The body closure can read the entry-block params slice in any order
//    and refer to the module to declare additional imports mid-body.
// ------------------------------------------------------------------

#[jit_export]
fn add_three(a: i64, b: i64, c: i64) -> i64 {
    a + b + c
}

#[test]
fn body_can_use_module_to_declare_imports_mid_function() {
    let mut jb = jit_builder();
    add_three_jit::register(&mut jb);
    let mut module = JITModule::new(jb);

    // Note: we don't pre-`declare` add_three; the closure does it on demand.
    let id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn(i64, i64, i64) -> i64,
        |bcx, module, params| {
            let ext = add_three_jit::declare(module);
            add_three_jit::call(bcx, module, ext, params[2], params[1], params[0])
        },
    )
    .unwrap();

    module.finalize_definitions().unwrap();
    let f: extern "C" fn(i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    assert_eq!(f(1, 2, 3), 6);
}
