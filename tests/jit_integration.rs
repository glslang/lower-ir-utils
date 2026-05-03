//! End-to-end JIT tests: declare an external Rust function, build a Cranelift
//! function that calls it via the macros, JIT-compile, and invoke the result.

use std::collections::HashMap;

use cranelift_codegen::ir::{InstBuilder, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use lower_ir_utils::{jit_call, jit_signature};

/// Build a JIT module with a single registered host symbol.
fn jit_with_symbol(name: &'static str, addr: *const u8) -> JITModule {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().unwrap();
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
    jb.symbol(name, addr);
    JITModule::new(jb)
}

// ------------------------------------------------------------------
// Test 1: i64 -> i64 (Value passthrough as the only argument).
// ------------------------------------------------------------------

extern "C" fn double_i64(x: i64) -> i64 {
    x.wrapping_mul(2)
}

#[test]
fn calls_extern_taking_i64() {
    let mut module = jit_with_symbol("double_i64", double_i64 as *const u8);

    let ext_sig = jit_signature!(&module; fn(i64) -> i64);
    let ext_id = module
        .declare_function("double_i64", Linkage::Import, &ext_sig)
        .unwrap();

    let wrap_sig = jit_signature!(&module; fn(i64) -> i64);
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let ptr_ty = module.target_config().pointer_type();
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

        let ext_local = module.declare_func_in_func(ext_id, bcx.func);
        let call = jit_call!(&mut bcx, ptr_ty, ext_local; x);
        let ret = bcx.inst_results(call)[0];
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(wrap_id);
    let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };
    assert_eq!(f(21), 42);
    assert_eq!(f(-7), -14);
}

// ------------------------------------------------------------------
// Test 2: (*const HashMap, &str) -> i64. Exercises &'static str lowering
// (constant ptr + len) and *const T parameter shape.
// ------------------------------------------------------------------

extern "C" fn lookup(
    map_ptr: *const HashMap<String, i64>,
    key_ptr: *const u8,
    key_len: usize,
) -> i64 {
    let map = unsafe { &*map_ptr };
    let key =
        unsafe { std::str::from_utf8(std::slice::from_raw_parts(key_ptr, key_len)).unwrap() };
    *map.get(key).unwrap_or(&-1)
}

#[test]
fn calls_extern_with_map_pointer_and_static_str() {
    let mut module = jit_with_symbol("lookup", lookup as *const u8);

    let ext_sig = jit_signature!(&module; fn(*const HashMap<String, i64>, &str) -> i64);
    // Sanity: 3 params (one ptr, two for &str) and one return.
    assert_eq!(ext_sig.params.len(), 3);
    assert_eq!(ext_sig.returns.len(), 1);

    let ext_id = module
        .declare_function("lookup", Linkage::Import, &ext_sig)
        .unwrap();

    // The wrapper takes the map pointer dynamically so we can inject the live map.
    let wrap_sig = jit_signature!(&module; fn(*const HashMap<String, i64>) -> i64);
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let ptr_ty = module.target_config().pointer_type();
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

        let ext_local = module.declare_func_in_func(ext_id, bcx.func);
        // map_v: passthrough Value; "answer": lowered as 2 iconsts.
        let call = jit_call!(&mut bcx, ptr_ty, ext_local; map_v, "answer");
        let ret = bcx.inst_results(call)[0];
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(wrap_id);
    let f: extern "C" fn(*const HashMap<String, i64>) -> i64 =
        unsafe { std::mem::transmute(code_ptr) };

    let mut map = HashMap::new();
    map.insert("answer".to_string(), 42i64);
    map.insert("other".to_string(), 7);
    assert_eq!(f(&map), 42);

    map.remove("answer");
    assert_eq!(f(&map), -1);
}

// ------------------------------------------------------------------
// Test 3: Mixed primitives — (i32, f64) -> f64. Verifies float lowering
// and that integer + float types produce the correct AbiParams.
// ------------------------------------------------------------------

extern "C" fn fma_like(n: i32, x: f64) -> f64 {
    (n as f64) * x + 1.0
}

#[test]
fn calls_extern_with_mixed_int_float() {
    let mut module = jit_with_symbol("fma_like", fma_like as *const u8);

    let ext_sig = jit_signature!(&module; fn(i32, f64) -> f64);
    let ext_id = module
        .declare_function("fma_like", Linkage::Import, &ext_sig)
        .unwrap();

    let wrap_sig = jit_signature!(&module; fn(i32, f64) -> f64);
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let ptr_ty = module.target_config().pointer_type();
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

        let ext_local = module.declare_func_in_func(ext_id, bcx.func);
        let call = jit_call!(&mut bcx, ptr_ty, ext_local; n, x);
        let ret = bcx.inst_results(call)[0];
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(wrap_id);
    let f: extern "C" fn(i32, f64) -> f64 = unsafe { std::mem::transmute(code_ptr) };
    assert_eq!(f(3, 0.5), 3.0 * 0.5 + 1.0);
}

// ------------------------------------------------------------------
// Test 4: Constant-pointer lowering. The wrapper takes no args; the address
// of a static struct is embedded into the IR via JitArg for *const T.
// ------------------------------------------------------------------

#[repr(C)]
struct Config {
    base: i64,
}

extern "C" fn add_to_base(cfg: *const Config, x: i64) -> i64 {
    let cfg = unsafe { &*cfg };
    cfg.base + x
}

static CFG: Config = Config { base: 100 };

#[test]
fn embeds_raw_pointer_constant() {
    let mut module = jit_with_symbol("add_to_base", add_to_base as *const u8);

    let ext_sig = jit_signature!(&module; fn(*const Config, i64) -> i64);
    let ext_id = module
        .declare_function("add_to_base", Linkage::Import, &ext_sig)
        .unwrap();

    let wrap_sig = jit_signature!(&module; fn(i64) -> i64);
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let ptr_ty = module.target_config().pointer_type();
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

        let ext_local = module.declare_func_in_func(ext_id, bcx.func);
        let cfg_ptr: *const Config = &CFG;
        let call = jit_call!(&mut bcx, ptr_ty, ext_local; cfg_ptr, x);
        let ret = bcx.inst_results(call)[0];
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(wrap_id);
    let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_ptr) };
    assert_eq!(f(5), 105);
    assert_eq!(f(-50), 50);
}

// ------------------------------------------------------------------
// Test 5: Zero-argument call. Verifies the macro handles the empty case.
// ------------------------------------------------------------------

extern "C" fn answer() -> i64 {
    42
}

#[test]
fn calls_extern_with_no_args() {
    let mut module = jit_with_symbol("answer", answer as *const u8);

    let ext_sig = jit_signature!(&module; fn() -> i64);
    assert!(ext_sig.params.is_empty());
    let ext_id = module
        .declare_function("answer", Linkage::Import, &ext_sig)
        .unwrap();

    let wrap_sig = jit_signature!(&module; fn() -> i64);
    let wrap_id = module
        .declare_function("wrap", Linkage::Export, &wrap_sig)
        .unwrap();

    let ptr_ty = module.target_config().pointer_type();
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

        let ext_local = module.declare_func_in_func(ext_id, bcx.func);
        let call = jit_call!(&mut bcx, ptr_ty, ext_local;);
        let ret = bcx.inst_results(call)[0];
        bcx.ins().return_(&[ret]);
        bcx.finalize();
    }

    module.define_function(wrap_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(wrap_id);
    let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(code_ptr) };
    assert_eq!(f(), 42);
}
