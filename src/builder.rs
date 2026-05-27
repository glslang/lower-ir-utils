//! High-level helper to define a Cranelift function: declares the function,
//! creates the `FunctionBuilderContext` + entry block, runs the user-supplied
//! body closure to populate the IR, emits the return, and finalizes.
//!
//! The body closure receives `&mut FunctionBuilder`, `&mut Module`, and the
//! entry-block parameters. It returns the values to be returned, via the
//! [`IntoReturns`] trait — so `()`, a single `Value`, `[Value; N]`, or
//! `Vec<Value>` all work without ceremony.

use cranelift_codegen::Context;
use cranelift_codegen::ir::{InstBuilder, Signature, UserFuncName, Value};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module, ModuleResult};
use smallvec::SmallVec;

/// Conversion from common Rust shapes to a return-Value list.
///
/// Used by [`define_function`] to let the body closure return whatever is most
/// natural: `()` for void, a bare `Value` for one return, `[Value; N]` /
/// `Vec<Value>` / `SmallVec` for multi-return.
pub trait IntoReturns {
    fn into_returns(self) -> SmallVec<[Value; 4]>;
}

impl IntoReturns for () {
    fn into_returns(self) -> SmallVec<[Value; 4]> {
        SmallVec::new()
    }
}

impl IntoReturns for Value {
    fn into_returns(self) -> SmallVec<[Value; 4]> {
        let mut s = SmallVec::new();
        s.push(self);
        s
    }
}

impl<const N: usize> IntoReturns for [Value; N] {
    fn into_returns(self) -> SmallVec<[Value; 4]> {
        SmallVec::from_iter(self)
    }
}

impl IntoReturns for Vec<Value> {
    fn into_returns(self) -> SmallVec<[Value; 4]> {
        SmallVec::from_vec(self)
    }
}

impl IntoReturns for SmallVec<[Value; 4]> {
    fn into_returns(self) -> SmallVec<[Value; 4]> {
        self
    }
}

/// Declare and define a Cranelift function in one call.
///
/// `body` is invoked with the live `FunctionBuilder`, the module (re-borrowed
/// so the body can declare imports / call other JITed functions), and the
/// entry block's parameters. Whatever it returns is wrapped via [`IntoReturns`]
/// and emitted as the function's `return_` instruction.
///
/// # Example
///
/// ```ignore
/// let id = define_function(
///     &mut module,
///     "wrap",
///     Linkage::Export,
///     jit_signature!(&module; fn(i64) -> i64),
///     |bcx, module, params| {
///         double_i64_jit::call(bcx, module, ext_id, params[0])
///     },
/// )?;
/// ```
#[allow(clippy::result_large_err)] // ModuleError is cranelift's type; not our concern.
pub fn define_function<M, F, R>(
    module: &mut M,
    name: &str,
    linkage: Linkage,
    signature: Signature,
    body: F,
) -> ModuleResult<FuncId>
where
    M: Module,
    F: FnOnce(&mut FunctionBuilder<'_>, &mut M, &[Value]) -> R,
    R: IntoReturns,
{
    let (func_id, mut ctx) = declare_and_build(module, name, linkage, signature, body)?;
    module.define_function(func_id, &mut ctx)?;
    module.clear_context(&mut ctx);
    Ok(func_id)
}

/// Shared inner step for [`define_function`] and its disasm-capturing variant.
///
/// Declares the function on `module`, builds a fresh `Context` with an entry
/// block containing the lowered body, but **does not** call
/// `module.define_function`. The caller drives the final compile step so it
/// can configure compile-time options (e.g. enabling disassembly capture)
/// before lowering happens.
#[allow(clippy::result_large_err)]
pub(crate) fn declare_and_build<M, F, R>(
    module: &mut M,
    name: &str,
    linkage: Linkage,
    signature: Signature,
    body: F,
) -> ModuleResult<(FuncId, Context)>
where
    M: Module,
    F: FnOnce(&mut FunctionBuilder<'_>, &mut M, &[Value]) -> R,
    R: IntoReturns,
{
    let func_id = module.declare_function(name, linkage, &signature)?;

    let mut ctx = module.make_context();
    ctx.func.signature = signature;
    ctx.func.name = UserFuncName::user(0, func_id.as_u32());

    let mut bcx_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut bcx_ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let params: SmallVec<[Value; 8]> = bcx.block_params(entry).iter().copied().collect();
        let returns = body(&mut bcx, module, &params).into_returns();
        bcx.ins().return_(&returns);
        bcx.finalize();
    }

    Ok((func_id, ctx))
}
