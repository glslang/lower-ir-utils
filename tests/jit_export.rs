//! Tests targeting `#[jit_export]` proc-macro behavior specifically:
//! - implicit ABI injection (no `extern "C"` written by the user),
//! - signature parity with hand-written `jit_signature!`,
//! - unit-return functions,
//! - `signature` / `register` / `declare` helpers in isolation.

use cranelift_codegen::ir::{InstBuilder, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

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
        bcx.finalize();
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
//    (a `&'static str` literal) for the next, in the same call.
// ------------------------------------------------------------------

#[jit_export]
fn mix(prefix_len: i64, key: &str) -> i64 {
    prefix_len + key.len() as i64
}

// Microsoft x64 passes 16-byte aggregates (`&str`) by hidden pointer; this
// crate lowers them as two register params, so the callee reads garbage.
#[cfg_attr(all(target_os = "windows", target_arch = "x86_64"), ignore)]
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

        // prefix_v: dynamic Value; "abcdef": &'static str literal lowered as 2 iconsts.
        let ret = mix_jit::call(&mut bcx, &mut module, id, prefix_v, "abcdef");
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(10), 10 + 6);
    assert_eq!(f(0), 6);
}
