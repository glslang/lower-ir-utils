//! End-to-end test for the `tokio` feature: compile a JIT function off the
//! async workers via `spawn_blocking_build`, then call the resulting
//! `extern "C" fn` pointer from async code (across an `.await`).

#![cfg(feature = "tokio")]

use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};

use lower_ir_utils::{define_jit_fn, jit_export, spawn_blocking_build};

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

#[jit_export]
fn double_i64(x: i64) -> i64 {
    x.wrapping_mul(2)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compiles_off_thread_and_calls_from_async() {
    let mut jb = jit_builder();
    double_i64_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = double_i64_jit::declare(&mut module);

    // Blocking compile runs on tokio's blocking pool; the module comes back so
    // we can pull the finalized pointer out on the async side.
    let (module, wrap_id) = spawn_blocking_build(module, move |m| {
        let id = define_jit_fn!(
            m,
            "wrap",
            Linkage::Export,
            fn(i64) -> i64,
            |bcx, m, params| double_i64_jit::call(bcx, m, ext_id, params[0]),
        )
        .unwrap();
        m.finalize_definitions().unwrap();
        id
    })
    .await;

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };

    assert_eq!(f(21), 42);

    // The pointer survives an await point and stays callable while `module` lives.
    tokio::task::yield_now().await;
    assert_eq!(f(-7), -14);
}

// An async function — JIT IR cannot call this directly (it has no executor and
// `async fn` cannot be `#[jit_export]`ed).
async fn fetch(id: i64) -> i64 {
    tokio::task::yield_now().await;
    id.wrapping_mul(10)
}

// The supported bridge: a *synchronous* `extern "C"` shim the JIT can call,
// which drives the async future to completion on the host.
#[jit_export]
fn fetch_sync(id: i64) -> i64 {
    tokio::runtime::Handle::current().block_on(fetch(id))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn jit_calls_async_via_sync_shim() {
    let mut jb = jit_builder();
    fetch_sync_jit::register(&mut jb);
    let mut module = JITModule::new(jb);
    let ext_id = fetch_sync_jit::declare(&mut module);

    let (module, wrap_id) = spawn_blocking_build(module, move |m| {
        let id = define_jit_fn!(
            m,
            "wrap_fetch",
            Linkage::Export,
            fn(i64) -> i64,
            |bcx, m, params| fetch_sync_jit::call(bcx, m, ext_id, params[0]),
        )
        .unwrap();
        m.finalize_definitions().unwrap();
        id
    })
    .await;

    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };

    // Call the JIT fn on a blocking-pool thread: the shim's `block_on` would
    // panic on an async worker, but a `spawn_blocking` thread is fine.
    let result = tokio::task::spawn_blocking(move || f(5)).await.unwrap();
    assert_eq!(result, 50);

    drop(module); // keep the executable memory alive until after the call
}
