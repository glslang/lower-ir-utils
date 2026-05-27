//! Async-runtime helpers (feature `tokio`).
//!
//! Cranelift compilation — chiefly [`JITModule::finalize_definitions`] and the
//! per-function `module.define_function` that [`define_function`] /
//! [`define_jit_fn!`] drive — is a synchronous, CPU-bound, blocking step.
//! Running it directly on a tokio worker thread stalls the async executor.
//! [`spawn_blocking_build`] moves that work onto [`tokio::task::spawn_blocking`]
//! and hands the module back so finalized function pointers can be extracted on
//! the async side, where calling them is cheap.
//!
//! [`JITModule::finalize_definitions`]: cranelift_jit::JITModule::finalize_definitions
//! [`define_function`]: crate::define_function
//! [`define_jit_fn!`]: crate::define_jit_fn

use cranelift_module::Module;

// `spawn_blocking` requires the moved value be `Send + 'static`. This guard
// fails to compile if a future Cranelift release drops that bound on
// `JITModule`, which would invalidate `spawn_blocking_build` for the JIT module.
const _: fn() = || {
    fn assert_send_static<T: Send + 'static>() {}
    assert_send_static::<cranelift_jit::JITModule>();
};

/// Run a blocking module build/finalize step on tokio's blocking thread pool.
///
/// Takes ownership of `module`, runs `f` (which typically declares/defines
/// functions and then calls `finalize_definitions()`) inside
/// [`tokio::task::spawn_blocking`], and returns the module alongside `f`'s
/// result. Extract finalized function pointers *after* the await — calling an
/// `extern "C" fn` obtained from the module does not need the blocking pool.
///
/// The returned module owns the executable memory and **must outlive every
/// call** to any function pointer obtained from it.
///
/// # Panics
///
/// Propagates a panic from `f` (the `spawn_blocking` task) into the caller.
///
/// # Example
///
/// ```ignore
/// let (module, wrap_id) = spawn_blocking_build(module, |m| {
///     let id = define_jit_fn!(m, "wrap", Linkage::Export, fn(i64) -> i64,
///         |bcx, m, p| double_i64_jit::call(bcx, m, ext_id, p[0])).unwrap();
///     m.finalize_definitions().unwrap();
///     id
/// })
/// .await;
///
/// let f: extern "C" fn(i64) -> i64 =
///     unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
/// assert_eq!(f(21), 42);
/// // `module` must stay alive while `f` is callable.
/// ```
pub async fn spawn_blocking_build<M, F, R>(mut module: M, f: F) -> (M, R)
where
    M: Module + Send + 'static,
    F: FnOnce(&mut M) -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let r = f(&mut module);
        (module, r)
    })
    .await
    .expect("JIT build task panicked")
}
