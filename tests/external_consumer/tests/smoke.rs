//! End-to-end JIT exercise from a crate that doesn't directly depend on
//! cranelift_jit / _module / _codegen / _frontend / smallvec. Cranelift types
//! are reached via `lower_ir_utils::__reexport`. cranelift-native is a
//! dev-dep here only to build the host ISA — the proc-macro itself never
//! generates paths into it.

use lower_ir_utils::__reexport::{
    cranelift_codegen::settings::{self, Configurable},
    cranelift_jit::{JITBuilder, JITModule},
    cranelift_module::{default_libcall_names, Linkage},
};
use lower_ir_utils::define_jit_fn;

use external_consumer::{add, add_jit, lookup_len_jit, record_jit};

fn host_isa() -> std::sync::Arc<dyn lower_ir_utils::__reexport::cranelift_codegen::isa::TargetIsa>
{
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap()
}

#[test]
fn name_constants_are_reachable() {
    assert_eq!(add_jit::NAME, "add");
    assert_eq!(lookup_len_jit::NAME, "lookup_len");
    assert_eq!(record_jit::NAME, "record");
}

#[test]
fn add_compiles_and_runs() {
    let mut jb = JITBuilder::with_isa(host_isa(), default_libcall_names());
    add_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = add_jit::declare(&mut module);

    let wrap_id = define_jit_fn!(
        &mut module, "wrap", Linkage::Export, fn(i64, i64) -> i64,
        |bcx, module, params| add_jit::call(bcx, module, ext_id, params[0], params[1]),
    )
    .unwrap();

    module.finalize_definitions().unwrap();
    let f: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(2, 3), 5);
    // Host-side direct call still works (the macro keeps the original fn).
    assert_eq!(add(2, 3), 5);
}

#[test]
fn str_param_compiles_and_runs() {
    let mut jb = JITBuilder::with_isa(host_isa(), default_libcall_names());
    lookup_len_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = lookup_len_jit::declare(&mut module);

    let wrap_id = define_jit_fn!(
        &mut module, "wrap", Linkage::Export, fn() -> i64,
        |bcx, module, _params| lookup_len_jit::call(bcx, module, ext_id, "external"),
    )
    .unwrap();

    module.finalize_definitions().unwrap();
    let f: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
    assert_eq!(f(), 8);
}
