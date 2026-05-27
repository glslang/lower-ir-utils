//! Property tests for the `sim` IR interpreter (`src/sim.rs`).
//!
//! A fuzzer over this module would need to synthesize Cranelift IR — and since
//! Cranelift's IR types don't implement `arbitrary::Arbitrary`, the bulk of the
//! work is a generator that builds well-formed `Function`s via
//! `FunctionBuilder`. This file is that generator, driven by `proptest` so it
//! runs on stable in the normal test suite (see the plan note on why this beats
//! standing up a `cargo fuzz` target for a structured-input interpreter).
//!
//! Two properties share one expression-AST generator:
//!
//! 1. **Differential correctness** ([`sim_matches_jit`]): for the trap-free
//!    op subset, the simulator's result must equal what the real JIT computes
//!    for the same IR and inputs. This is the ground-truth oracle.
//! 2. **Robustness** ([`div_never_panics`], [`load_never_panics`],
//!    [`store_never_panics`]): for the risky paths a fuzzer hunts in — division
//!    by zero / `INT_MIN / -1`, and out-of-bounds memory — the simulator must
//!    never panic; it must return cleanly or surface a known [`SimError`].
//!
//! Generated IR is a straight-line DAG (no blocks/branches), so these tests
//! never build a loop — but `Simulator::run` is independently bounded by
//! `Simulator::max_steps` (`src/sim.rs`), so even a back-edge would halt with
//! `SimError::StepLimitExceeded` rather than hang.

#![cfg(feature = "sim")]

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types, AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use lower_ir_utils::define_function;
use lower_ir_utils::sim::{SimError, SimValue, Simulator};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Expression AST. Every node evaluates to an i64, which keeps the generated
// IR composable and the differential comparison a single scalar.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Ushr,
    Sshr,
    // Trapping in real Cranelift (div-by-zero, INT_MIN/-1); generated only for
    // the sim-only robustness test, never for the differential one.
    Sdiv,
    Udiv,
    Srem,
    Urem,
}

#[derive(Clone, Debug)]
enum Expr {
    Param(usize),
    Const(i64),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    AddImm(Box<Expr>, i64),
    MulImm(Box<Expr>, i64),
    Neg(Box<Expr>),
    Bnot(Box<Expr>),
    /// `icmp` (yields I8 0/1) then `uextend` to I64, so compares compose.
    Cmp(IntCC, Box<Expr>, Box<Expr>),
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
}

const NUM_PARAMS: usize = 2;

fn arb_binop(allow_div: bool) -> impl Strategy<Value = BinOp> {
    let mut ops = vec![
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::Ushr,
        BinOp::Sshr,
    ];
    if allow_div {
        ops.extend([BinOp::Sdiv, BinOp::Udiv, BinOp::Srem, BinOp::Urem]);
    }
    proptest::sample::select(ops)
}

fn arb_intcc() -> impl Strategy<Value = IntCC> {
    proptest::sample::select(vec![
        IntCC::Equal,
        IntCC::NotEqual,
        IntCC::SignedLessThan,
        IntCC::SignedGreaterThan,
        IntCC::SignedLessThanOrEqual,
        IntCC::SignedGreaterThanOrEqual,
        IntCC::UnsignedLessThan,
        IntCC::UnsignedGreaterThan,
        IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedGreaterThanOrEqual,
    ])
}

fn arb_expr(allow_div: bool) -> impl Strategy<Value = Expr> {
    let leaf = prop_oneof![
        (0..NUM_PARAMS).prop_map(Expr::Param),
        any::<i64>().prop_map(Expr::Const),
    ];
    // depth 5, ~48 nodes target, ~3 children per branch.
    leaf.prop_recursive(5, 48, 3, move |inner| {
        prop_oneof![
            (arb_binop(allow_div), inner.clone(), inner.clone()).prop_map(|(op, a, b)| Expr::Bin(
                op,
                Box::new(a),
                Box::new(b)
            )),
            (inner.clone(), any::<i64>()).prop_map(|(a, i)| Expr::AddImm(Box::new(a), i)),
            (inner.clone(), any::<i64>()).prop_map(|(a, i)| Expr::MulImm(Box::new(a), i)),
            inner.clone().prop_map(|a| Expr::Neg(Box::new(a))),
            inner.clone().prop_map(|a| Expr::Bnot(Box::new(a))),
            (arb_intcc(), inner.clone(), inner.clone()).prop_map(|(cc, a, b)| Expr::Cmp(
                cc,
                Box::new(a),
                Box::new(b)
            )),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(|(c, a, b)| Expr::Select(
                Box::new(c),
                Box::new(a),
                Box::new(b)
            )),
        ]
    })
}

/// Lower an [`Expr`] into IR under `bcx`, returning the root i64 `Value`.
/// Shared by the JIT path and the simulator path so both interpret identical
/// IR.
fn build(bcx: &mut FunctionBuilder, params: &[Value], e: &Expr) -> Value {
    match e {
        Expr::Param(i) => params[i % params.len()],
        Expr::Const(c) => bcx.ins().iconst(types::I64, *c),
        Expr::Bin(op, a, b) => {
            let x = build(bcx, params, a);
            let y = build(bcx, params, b);
            match op {
                BinOp::Add => bcx.ins().iadd(x, y),
                BinOp::Sub => bcx.ins().isub(x, y),
                BinOp::Mul => bcx.ins().imul(x, y),
                BinOp::And => bcx.ins().band(x, y),
                BinOp::Or => bcx.ins().bor(x, y),
                BinOp::Xor => bcx.ins().bxor(x, y),
                BinOp::Shl => bcx.ins().ishl(x, y),
                BinOp::Ushr => bcx.ins().ushr(x, y),
                BinOp::Sshr => bcx.ins().sshr(x, y),
                BinOp::Sdiv => bcx.ins().sdiv(x, y),
                BinOp::Udiv => bcx.ins().udiv(x, y),
                BinOp::Srem => bcx.ins().srem(x, y),
                BinOp::Urem => bcx.ins().urem(x, y),
            }
        }
        Expr::AddImm(a, imm) => {
            let x = build(bcx, params, a);
            bcx.ins().iadd_imm(x, *imm)
        }
        Expr::MulImm(a, imm) => {
            let x = build(bcx, params, a);
            bcx.ins().imul_imm(x, *imm)
        }
        Expr::Neg(a) => {
            let x = build(bcx, params, a);
            bcx.ins().ineg(x)
        }
        Expr::Bnot(a) => {
            let x = build(bcx, params, a);
            bcx.ins().bnot(x)
        }
        Expr::Cmp(cc, a, b) => {
            let x = build(bcx, params, a);
            let y = build(bcx, params, b);
            let c = bcx.ins().icmp(*cc, x, y); // I8 (0/1)
            bcx.ins().uextend(types::I64, c)
        }
        Expr::Select(c, a, b) => {
            let cond = build(bcx, params, c);
            let x = build(bcx, params, a);
            let y = build(bcx, params, b);
            bcx.ins().select(cond, x, y)
        }
    }
}

// ---------------------------------------------------------------------------
// Building standalone `Function`s for the simulator.
// ---------------------------------------------------------------------------

fn empty_func(params: &[types::Type], returns: &[types::Type]) -> Function {
    let mut sig = Signature::new(CallConv::Fast);
    for p in params {
        sig.params.push(AbiParam::new(*p));
    }
    for r in returns {
        sig.returns.push(AbiParam::new(*r));
    }
    Function::with_name_signature(UserFuncName::default(), sig)
}

/// Build a single-block `fn(i64, i64) -> i64` whose body is `e`.
fn build_sim_func(e: &Expr) -> Function {
    let mut func = empty_func(&[types::I64, types::I64], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let params: Vec<Value> = bcx.block_params(entry).to_vec();
        let v = build(&mut bcx, &params, e);
        bcx.ins().return_(&[v]);
        bcx.finalize();
    }
    func
}

/// JIT-compile and run `fn(i64, i64) -> i64` with body `e` — the ground-truth
/// oracle. Mirrors the setup in `tests/jit_integration.rs`.
fn jit_eval(e: &Expr, a: i64, b: i64) -> i64 {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let jb = JITBuilder::with_isa(isa, default_libcall_names());
    let mut module = JITModule::new(jb);

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));

    let id = define_function(&mut module, "f", Linkage::Export, sig, |bcx, _m, params| {
        let p = params.to_vec();
        build(bcx, &p, e)
    })
    .unwrap();

    module.finalize_definitions().unwrap();
    // FFI test glue, per CLAUDE.md: transmute the finalized pointer and call it
    // before the module (which owns the code) is dropped.
    let code = module.get_finalized_function(id);
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
    f(a, b)
}

/// `load.i64 base+offset; return` — exercises the simulator's bounds checking.
fn build_load_func(offset: i32) -> Function {
    let mut func = empty_func(&[types::I64], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let base = bcx.block_params(entry)[0];
        let loaded = bcx.ins().load(types::I64, MemFlags::new(), base, offset);
        bcx.ins().return_(&[loaded]);
        bcx.finalize();
    }
    func
}

/// `store.i64 0xA5A5..., base+offset; return` — exercises store bounds checking.
fn build_store_func(offset: i32) -> Function {
    let mut func = empty_func(&[types::I64], &[]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let base = bcx.block_params(entry)[0];
        let val = bcx
            .ins()
            .iconst(types::I64, 0xA5A5_A5A5_A5A5_A5A5_u64 as i64);
        bcx.ins().store(MemFlags::new(), val, base, offset);
        bcx.ins().return_(&[]);
        bcx.finalize();
    }
    func
}

fn err_is_known(err: &SimError) -> bool {
    matches!(
        err,
        SimError::UnsupportedOpcode(_)
            | SimError::OutOfBounds { .. }
            | SimError::UndefinedValue(_)
            | SimError::TypeMismatch(_)
    )
}

proptest! {
    // JIT-compiling per case is the slow path; keep the count modest.
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// The simulator's result must match the real JIT for the trap-free op set.
    #[test]
    fn sim_matches_jit(e in arb_expr(false), a in any::<i64>(), b in any::<i64>()) {
        let expected = jit_eval(&e, a, b);

        let func = build_sim_func(&e);
        let mut sim = Simulator::new(0);
        let res = sim.run(&func, &[SimValue::I64(a), SimValue::I64(b)]);

        prop_assert!(res.error.is_none(), "sim halted: {:?}", res.error);
        prop_assert_eq!(res.returns.len(), 1);
        prop_assert_eq!(res.returns[0].as_i64(), expected);
    }
}

proptest! {
    // Pure-sim cases are cheap; sweep more of them.
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// Division (incl. div-by-zero and `INT_MIN / -1`) must never panic the
    /// simulator. Not compared against the JIT, which *traps* on those.
    #[test]
    fn div_never_panics(e in arb_expr(true), a in any::<i64>(), b in any::<i64>()) {
        let func = build_sim_func(&e);
        let mut sim = Simulator::new(0);
        let res = sim.run(&func, &[SimValue::I64(a), SimValue::I64(b)]);
        if let Some(err) = &res.error {
            prop_assert!(err_is_known(err), "unexpected error: {err:?}");
        }
    }

    /// A `load` at any base/offset against any-sized memory must either
    /// succeed or report `OutOfBounds` — never panic or read OOB.
    #[test]
    fn load_never_panics(mem_len in 0usize..256, base in any::<i64>(), offset in -128i32..128) {
        let func = build_load_func(offset);
        let mut sim = Simulator::with_memory(vec![0u8; mem_len]);
        let res = sim.run(&func, &[SimValue::I64(base)]);
        if let Some(err) = &res.error {
            prop_assert!(matches!(err, SimError::OutOfBounds { .. }), "got {err:?}");
        }
    }

    /// Symmetric guarantee for `store`.
    #[test]
    fn store_never_panics(mem_len in 0usize..256, base in any::<i64>(), offset in -128i32..128) {
        let func = build_store_func(offset);
        let before = vec![0u8; mem_len];
        let mut sim = Simulator::with_memory(before.clone());
        let res = sim.run(&func, &[SimValue::I64(base)]);
        if let Some(err) = &res.error {
            prop_assert!(matches!(err, SimError::OutOfBounds { .. }), "got {err:?}");
            // On an out-of-bounds store, memory must be left untouched.
            prop_assert_eq!(&res.memory, &before);
        }
    }
}
