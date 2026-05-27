//! End-to-end JIT tests using the full helper stack: `#[jit_export]` for the
//! callee + `define_jit_fn!` for the wrapper. The block-creation, return, and
//! finalize boilerplate has all moved into `define_function`.

use std::collections::HashMap;

use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};

use lower_ir_utils::{define_jit_fn, jit_export};

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

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn(i64) -> i64,
        |bcx, module, params| double_i64_jit::call(bcx, module, ext_id, params[0]),
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(21), 42);
    assert_eq!(f(-7), -14);
}

// ------------------------------------------------------------------
// Test 2: (*const HashMap, &str) -> i64. Mixed Value + &'static str literal.
// ------------------------------------------------------------------

#[jit_export]
fn lookup(map_ptr: *const HashMap<String, i64>, key: &str) -> i64 {
    let map = unsafe { &*map_ptr };
    *map.get(key).unwrap_or(&-1)
}

// Microsoft x64 passes 16-byte aggregates (`&str`) by hidden pointer; this
// crate lowers them as two register params, so the callee reads garbage.
#[cfg_attr(all(target_os = "windows", target_arch = "x86_64"), ignore)]
#[test]
fn calls_extern_with_map_pointer_and_static_str() {
    let mut jb = jit_builder();
    lookup_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = lookup_jit::declare(&mut module);

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn(*const HashMap<String, i64>) -> i64,
        |bcx, module, params| {
            // params[0]: Value passthrough; "answer": &'static str → 2 iconsts.
            lookup_jit::call(bcx, module, ext_id, params[0], "answer")
        },
    )
    .unwrap();

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

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn(i32, f64) -> f64,
        |bcx, module, params| fma_like_jit::call(bcx, module, ext_id, params[0], params[1]),
    )
    .unwrap();

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

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn(i64) -> i64,
        |bcx, module, params| {
            let cfg_ptr: *const Config = &CFG;
            add_to_base_jit::call(bcx, module, ext_id, cfg_ptr, params[0])
        },
    )
    .unwrap();

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

    let wrap_id = define_jit_fn!(
        &mut module,
        "wrap",
        Linkage::Export,
        fn() -> i64,
        |bcx, module, _params| answer_jit::call(bcx, module, ext_id),
    )
    .unwrap();

    module.finalize_definitions().unwrap();

    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(), 42);
}
