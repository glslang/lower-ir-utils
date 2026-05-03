//! End-to-end JIT tests rewritten to use the `#[jit_export]` proc-macro.
//!
//! Compare the line counts to git history to see the boilerplate reduction:
//! every test now does symbol registration + signature build + import declare
//! in two lines (`foo_jit::register` + `foo_jit::declare`), and replaces the
//! `declare_func_in_func` + `jit_call!` pair with a single `foo_jit::call`.

use std::collections::HashMap;

use cranelift_codegen::ir::{InstBuilder, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use lower_ir_utils::{jit_export, jit_signature};

/// Build a fresh `JITBuilder` with no host symbols registered.
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
// Test 1: i64 -> i64 — Value passthrough.
// ------------------------------------------------------------------

#[jit_export]
fn double_i64(x: i64) -> i64 {
    x.wrapping_mul(2)
}

#[test]
fn calls_extern_taking_i64() {
    let mut jb = jit_builder();
    double_i64_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = double_i64_jit::declare(&mut module);

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
        let x = bcx.block_params(entry)[0];

        let ret = double_i64_jit::call(&mut bcx, &mut module, ext_id, x);
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(21), 42);
    assert_eq!(f(-7), -14);
}

// ------------------------------------------------------------------
// Test 2: (*const HashMap, &str) -> i64. Mixed Value + &'static str literal.
// ------------------------------------------------------------------

// On x86_64 SystemV the ABI of `extern "C" fn(&str)` matches `(*const u8, usize)`,
// so we can take an idiomatic `&str` in the Rust signature. The lint warning on
// `extern "C"` + `&str` is suppressed by the macro for the user's convenience —
// on platforms with different aggregate-passing rules (e.g. Win64) prefer the
// flat `(*const u8, usize)` form.
#[jit_export]
fn lookup(map_ptr: *const HashMap<String, i64>, key: &str) -> i64 {
    let map = unsafe { &*map_ptr };
    *map.get(key).unwrap_or(&-1)
}

#[test]
fn calls_extern_with_map_pointer_and_static_str() {
    let mut jb = jit_builder();
    lookup_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = lookup_jit::declare(&mut module);

    let wrap_sig = jit_signature!(&module; fn(*const HashMap<String, i64>) -> i64);
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
        let map_v = bcx.block_params(entry)[0];

        // map_v: Value passthrough; "answer": &'static str lowered as 2 iconsts.
        let ret = lookup_jit::call(&mut bcx, &mut module, ext_id, map_v, "answer");
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(*const HashMap<String, i64>) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };

    let mut map = HashMap::new();
    map.insert("answer".to_string(), 42i64);
    map.insert("other".to_string(), 7);
    assert_eq!(f(&map), 42);

    map.remove("answer");
    assert_eq!(f(&map), -1);
}

// ------------------------------------------------------------------
// Test 3: (i32, f64) -> f64.
// ------------------------------------------------------------------

#[jit_export]
fn fma_like(n: i32, x: f64) -> f64 {
    (n as f64) * x + 1.0
}

#[test]
fn calls_extern_with_mixed_int_float() {
    let mut jb = jit_builder();
    fma_like_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = fma_like_jit::declare(&mut module);

    let wrap_sig = jit_signature!(&module; fn(i32, f64) -> f64);
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
        let n = bcx.block_params(entry)[0];
        let x = bcx.block_params(entry)[1];

        let ret = fma_like_jit::call(&mut bcx, &mut module, ext_id, n, x);
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i32, f64) -> f64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(3, 0.5), 3.0 * 0.5 + 1.0);
}

// ------------------------------------------------------------------
// Test 4: Constant-pointer lowering — pass a *const T at codegen time.
// ------------------------------------------------------------------

#[repr(C)]
struct Config {
    base: i64,
}

#[jit_export]
fn add_to_base(cfg: *const Config, x: i64) -> i64 {
    let cfg = unsafe { &*cfg };
    cfg.base + x
}

static CFG: Config = Config { base: 100 };

#[test]
fn embeds_raw_pointer_constant() {
    let mut jb = jit_builder();
    add_to_base_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = add_to_base_jit::declare(&mut module);

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
        let x = bcx.block_params(entry)[0];

        let cfg_ptr: *const Config = &CFG;
        let ret = add_to_base_jit::call(&mut bcx, &mut module, ext_id, cfg_ptr, x);
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(5), 105);
    assert_eq!(f(-50), 50);
}

// ------------------------------------------------------------------
// Test 5: Zero-argument call.
// ------------------------------------------------------------------

#[jit_export]
fn answer() -> i64 {
    42
}

#[test]
fn calls_extern_with_no_args() {
    let mut jb = jit_builder();
    answer_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = answer_jit::declare(&mut module);

    let wrap_sig = jit_signature!(&module; fn() -> i64);
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

        let ret = answer_jit::call(&mut bcx, &mut module, ext_id);
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(), 42);
}
