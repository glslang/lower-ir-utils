//! Tests targeting `#[jit_export]` proc-macro behavior specifically:
//! - implicit ABI injection (no `extern "C"` written by the user),
//! - signature parity with hand-written `jit_signature!`,
//! - unit-return functions,
//! - `signature` / `register` / `declare` helpers in isolation.

use cranelift_codegen::ir::{InstBuilder, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};

use lower_ir_utils::{jit_export, jit_signature};

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
// 1. The macro auto-injects `extern "C"` so the user can write `fn`.
//    If the macro didn't, transmuting the symbol_addr to extern "C"
//    fn pointer would be UB and the call below would misbehave.
// ------------------------------------------------------------------

#[jit_export]
fn no_explicit_abi(x: i64) -> i64 {
    x + 1000
}

#[test]
fn macro_auto_injects_extern_c() {
    // The generated symbol_addr should point at an extern "C" function.
    // Cast and call directly — if the abi were Rust this would be UB,
    // but since the macro injected extern "C", this is well-defined.
    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(no_explicit_abi_jit::symbol_addr()) };
    assert_eq!(f(42), 1042);
}

// ------------------------------------------------------------------
// 2. The signature produced by the macro matches the hand-written
//    `jit_signature!` for the same Rust types.
// ------------------------------------------------------------------

#[jit_export]
fn three_params(a: i32, b: f64, c: *const u8) -> i64 {
    let _ = (a, b, c);
    0
}

#[test]
fn generated_signature_matches_jit_signature_macro() {
    let mut jb = jit_builder();
    three_params_jit::register(&mut jb);
    let module = JITModule::new(jb);

    let from_macro = three_params_jit::signature(&module);
    let from_hand = jit_signature!(&module; fn(i32, f64, *const u8) -> i64);

    let to_types = |s: &cranelift_codegen::ir::Signature| {
        (
            s.params.iter().map(|p| p.value_type).collect::<Vec<_>>(),
            s.returns.iter().map(|p| p.value_type).collect::<Vec<_>>(),
        )
    };
    assert_eq!(to_types(&from_macro), to_types(&from_hand));
    assert_eq!(from_macro.call_conv, from_hand.call_conv);
}

// ------------------------------------------------------------------
// 3. Unit-return function — `call` returns `Inst` (since there's no Value),
//    and `signature.returns` is empty.
// ------------------------------------------------------------------

use std::sync::atomic::{AtomicI64, Ordering};

static SIDE_EFFECT: AtomicI64 = AtomicI64::new(0);

#[jit_export]
fn side_effect(x: i64) {
    SIDE_EFFECT.store(x, Ordering::SeqCst);
}

#[test]
fn unit_return_works() {
    let mut jb = jit_builder();
    side_effect_jit::register(&mut jb);
    let mut module = JITModule::new(jb);

    // `signature` should report zero returns.
    assert!(side_effect_jit::signature(&module).returns.is_empty());

    let id = side_effect_jit::declare(&mut module);

    let wrap_sig = jit_signature!(&module; fn(i64));
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let mut ctx = module.make_context();
    ctx.func.signature = wrap_sig;
    ctx.func.name = UserFuncName::user(0, wrap_id.as_u32());

    let mut bcx_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut bcx_ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let x = bcx.block_params(entry)[0];

        // For a unit-return function, `call` returns `Inst`, not `Value`.
        let _inst: cranelift_codegen::ir::Inst =
            side_effect_jit::call(&mut bcx, &mut module, id, x);
        bcx.ins().return_(&[]);
        bcx.finalize(module.target_config());
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i64) =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    SIDE_EFFECT.store(0, Ordering::SeqCst);
    f(7777);
    assert_eq!(SIDE_EFFECT.load(Ordering::SeqCst), 7777);
}

// ------------------------------------------------------------------
// 4. `NAME` constant exposes the symbol name for diagnostics / logging.
// ------------------------------------------------------------------

#[jit_export]
fn named_thing(_: i64) -> i64 {
    0
}

#[test]
fn name_constant_matches_fn_name() {
    assert_eq!(named_thing_jit::NAME, "named_thing");
}

// ------------------------------------------------------------------
// 5. Mixed args at the call site: per-position generics let us pass an
//    already-lowered `Value` for one position and a Rust constant
//    (a pointer and a length) for the next positions in the same call.
// ------------------------------------------------------------------

#[jit_export]
fn mix(prefix_len: i64, _key: *const u8, key_len: usize) -> i64 {
    prefix_len + key_len as i64
}

// Multiple host results use explicit caller-owned output storage.
#[jit_export]
fn divmod(a: i64, b: i64, out: &mut [i64; 2]) {
    *out = [a / b, a % b];
}

#[test]
fn multiple_results_use_output_pointer() {
    let mut jb = jit_builder();
    divmod_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = divmod_jit::declare(&mut module);
    let id = lower_ir_utils::define_jit_fn!(
        &mut module,
        "wrap_divmod",
        Linkage::Export,
        fn(i64, i64, &mut [i64; 2]),
        |bcx, module, p| {
            divmod_jit::call(bcx, module, ext_id, p[0], p[1], p[2]);
        },
    )
    .unwrap();
    module.finalize_definitions().unwrap();
    let f: extern "C" fn(i64, i64, &mut [i64; 2]) =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    let mut out = [0; 2];
    f(17, 5, &mut out);
    assert_eq!(out, [3, 2]);
    f(20, 4, &mut out);
    assert_eq!(out, [5, 0]);
}

// ------------------------------------------------------------------
// 7. `&'static T` is a `JitArg`: the host address is embedded as an IR
//    constant just like a raw pointer, but with a lifetime bound that
//    rules out stack/heap pointers.
// ------------------------------------------------------------------

static THE_NUMBER: i64 = 99;

#[jit_export]
fn deref_i64(p: &i64) -> i64 {
    *p
}

#[test]
fn static_ref_lowers_as_immediate_pointer() {
    let mut jb = jit_builder();
    deref_i64_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = deref_i64_jit::declare(&mut module);

    let wrap_sig = jit_signature!(&module; fn() -> i64);
    let wrap_id = module
        .declare_function("wrap_deref", Linkage::Export, &wrap_sig)
        .unwrap();

    let mut ctx = module.make_context();
    ctx.func.signature = wrap_sig;
    ctx.func.name = UserFuncName::user(0, wrap_id.as_u32());

    let mut bcx_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut bcx_ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);

        // &'static i64 lowered as a single iconst pointer immediate. The
        // reference type is load-bearing — passing `THE_NUMBER` by value would
        // pick `JitArg for i64` and embed the literal `99` instead of its
        // address.
        let static_ref: &'static i64 = &THE_NUMBER;
        let ret = deref_i64_jit::call(&mut bcx, &mut module, ext_id, static_ref);
        bcx.ins().return_(&[ret]);
        bcx.finalize(module.target_config());
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(), 99);
}

// ------------------------------------------------------------------
// 8. `try_declare` exposes the underlying ModuleResult so callers can
//    surface declare-time errors instead of panicking.
// ------------------------------------------------------------------

#[jit_export]
fn try_declare_target(x: i64) -> i64 {
    x + 1
}

#[test]
fn try_declare_succeeds_for_fresh_module() {
    let mut jb = jit_builder();
    try_declare_target_jit::register(&mut jb);
    let mut module = JITModule::new(jb);

    let id = try_declare_target_jit::try_declare(&mut module).unwrap();
    // declare() called after try_declare() should be idempotent (same id) —
    // both end up requesting the same Import declaration with the same sig.
    let id2 = try_declare_target_jit::declare(&mut module);
    assert_eq!(id, id2);
}

#[test]
fn try_declare_surfaces_signature_conflict() {
    let mut jb = jit_builder();
    try_declare_target_jit::register(&mut jb);
    let mut module = JITModule::new(jb);

    // Pre-declare the same name with a different signature — `try_declare`
    // must surface the cranelift error instead of panicking.
    let conflicting_sig = jit_signature!(&module; fn() -> i64);
    module
        .declare_function(
            try_declare_target_jit::NAME,
            Linkage::Import,
            &conflicting_sig,
        )
        .unwrap();

    let result = try_declare_target_jit::try_declare(&mut module);
    assert!(
        result.is_err(),
        "expected declare to fail on signature conflict"
    );
}

#[test]
fn mixed_value_and_literal_args() {
    let mut jb = jit_builder();
    mix_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let id = mix_jit::declare(&mut module);

    let wrap_sig = jit_signature!(&module; fn(i64) -> i64);
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let mut ctx = module.make_context();
    ctx.func.signature = wrap_sig;
    ctx.func.name = UserFuncName::user(0, wrap_id.as_u32());

    let mut bcx_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut bcx_ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let prefix_v = bcx.block_params(entry)[0];

        // A dynamic Value plus explicit pointer and length constants.
        let ret = mix_jit::call(
            &mut bcx,
            &mut module,
            id,
            prefix_v,
            "abcdef".as_ptr(),
            6usize,
        );
        bcx.ins().return_(&[ret]);
        bcx.finalize(module.target_config());
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(10), 10 + 6);
    assert_eq!(f(0), 6);
}
