/// Build a Cranelift `Signature` from Rust types.
///
/// The first argument is an expression evaluating to a `&impl cranelift_module::Module`
/// (the module supplies both the call convention and the target pointer type).
///
/// # Example
///
/// ```ignore
/// let sig = jit_signature!(&module; fn(*const HashMap<String, i64>, &str) -> i64);
/// ```
#[macro_export]
macro_rules! jit_signature {
    ($module:expr; fn($($pty:ty),* $(,)?) $(-> $ret:ty)?) => {{
        let __m = $module;
        let mut __sig = $crate::__reexport::cranelift_module::Module::make_signature(__m);
        let __ptr = $crate::__reexport::cranelift_module::Module::target_config(__m).pointer_type();
        $(
            <$pty as $crate::JitParam>::push_params(&mut __sig.params, __ptr);
        )*
        $(
            <$ret as $crate::JitParam>::push_params(&mut __sig.returns, __ptr);
        )?
        __sig
    }};
}

/// Declare and define a Cranelift function in one call, with the signature
/// expressed as Rust types.
///
/// Combines [`jit_signature!`] with [`crate::define_function`]: the signature
/// is derived from the `fn(T1, T2) -> R` shape, then the body closure is
/// invoked with `&mut FunctionBuilder`, `&mut Module`, and the entry-block
/// parameter slice. Whatever the closure returns is funneled through
/// [`crate::IntoReturns`] and emitted as `return_`.
///
/// # Example
///
/// ```ignore
/// let id = define_jit_fn!(
///     &mut module, "wrap", Linkage::Export, fn(i64) -> i64,
///     |bcx, module, params| {
///         double_i64_jit::call(bcx, module, ext_id, params[0])
///     },
/// )?;
/// ```
#[macro_export]
macro_rules! define_jit_fn {
    (
        $module:expr,
        $name:expr,
        $linkage:expr,
        fn($($pty:ty),* $(,)?) $(-> $ret:ty)?,
        $body:expr $(,)?
    ) => {{
        let __module: &mut _ = $module;
        let __sig = $crate::jit_signature!(&*__module; fn($($pty),*) $(-> $ret)?);
        $crate::define_function(__module, $name, $linkage, __sig, $body)
    }};
}

/// Emit a Cranelift `call` instruction, lowering each Rust argument via [`JitArg`].
///
/// Each argument expression must implement [`JitArg`]. Already-lowered IR `Value`s
/// pass through unchanged; Rust constants (integers, floats, `&'static str`, raw
/// pointers) are emitted as IR constants on the spot.
///
/// Returns the `Inst` produced by `bcx.ins().call(...)` so the caller can extract
/// return values via `bcx.inst_results(inst)`.
///
/// # Example
///
/// ```ignore
/// // map_v: Value from a function param; "foo": &'static str literal.
/// let call = jit_call!(&mut bcx, ptr_ty, local_callee; map_v, "foo");
/// let result = bcx.inst_results(call)[0];
/// ```
///
/// [`JitArg`]: crate::JitArg
#[macro_export]
macro_rules! jit_call {
    ($bcx:expr, $ptr_ty:expr, $callee:expr $(; $($arg:expr),* $(,)?)?) => {{
        let __bcx = &mut *$bcx;
        let __ptr_ty = $ptr_ty;
        let mut __args: $crate::__reexport::smallvec::SmallVec<
            [$crate::__reexport::cranelift_codegen::ir::Value; 8]
        > = $crate::__reexport::smallvec::SmallVec::new();
        $($(
            <_ as $crate::JitArg>::lower($arg, __bcx, __ptr_ty, &mut __args);
        )*)?
        __bcx.ins().call($callee, &__args)
    }};
}
