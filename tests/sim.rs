//! End-to-end tests for the `sim` feature: build small Cranelift IR
//! functions via `FunctionBuilder`, run them through `Simulator`, and
//! assert on returns, register state, and memory.

#![cfg(feature = "sim")]

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types, AbiParam, ExtFuncData, ExternalName, Function, InstBuilder, MemFlags, Signature,
    UserExternalName, UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

use lower_ir_utils::jit_call;
use lower_ir_utils::sim::{SimError, SimValue, Simulator};

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

// 1. iconst 42; return  →  ret[0] == I64(42)
#[test]
fn const_return() {
    let mut func = empty_func(&[], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let v = bcx.ins().iconst(types::I64, 42);
        bcx.ins().return_(&[v]);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    let result = sim.run(&func, &[]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I64(42)]);
}

// 2. (a + 1) * 2: verify register table + return
#[test]
fn arith_chain() {
    let mut func = empty_func(&[types::I64], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    let (after_add, after_mul);
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        after_add = bcx.ins().iadd_imm(a, 1);
        after_mul = bcx.ins().imul_imm(after_add, 2);
        bcx.ins().return_(&[after_mul]);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    let result = sim.run(&func, &[SimValue::I64(20)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I64(42)]);
    assert_eq!(result.registers.get(&after_add), Some(&SimValue::I64(21)));
    assert_eq!(result.registers.get(&after_mul), Some(&SimValue::I64(42)));
}

// 3. Pre-fill memory, load.i64 at +8, store result at +16.
#[test]
fn load_store() {
    let mut func = empty_func(&[types::I64], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let base = bcx.block_params(entry)[0];
        let loaded = bcx.ins().load(types::I64, MemFlags::new(), base, 8);
        bcx.ins().store(MemFlags::new(), loaded, base, 16);
        bcx.ins().return_(&[loaded]);
        bcx.finalize();
    }

    let mut mem = vec![0u8; 64];
    mem[8..16].copy_from_slice(&0xDEAD_BEEF_CAFE_BABE_u64.to_le_bytes());
    let mut sim = Simulator::with_memory(mem);
    let result = sim.run(&func, &[SimValue::I64(0)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(
        result.returns,
        vec![SimValue::I64(0xDEAD_BEEF_CAFE_BABE_u64 as i64)]
    );
    assert_eq!(
        &result.memory[16..24],
        &0xDEAD_BEEF_CAFE_BABE_u64.to_le_bytes()
    );
}

// 4. if a > b { a } else { b } via icmp + brif + block params.
#[test]
fn branch_max() {
    fn build() -> Function {
        let mut func = empty_func(&[types::I64, types::I64], &[types::I64]);
        let mut ctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
            let entry = bcx.create_block();
            let exit = bcx.create_block();
            bcx.append_block_params_for_function_params(entry);
            bcx.append_block_param(exit, types::I64);
            bcx.switch_to_block(entry);
            bcx.seal_block(entry);
            let a = bcx.block_params(entry)[0];
            let b = bcx.block_params(entry)[1];
            let cmp = bcx.ins().icmp(IntCC::SignedGreaterThan, a, b);
            bcx.ins().brif(cmp, exit, &[a.into()], exit, &[b.into()]);
            bcx.seal_block(exit);
            bcx.switch_to_block(exit);
            let r = bcx.block_params(exit)[0];
            bcx.ins().return_(&[r]);
            bcx.finalize();
        }
        func
    }

    let func = build();
    let mut sim = Simulator::new(0);
    let r1 = sim.run(&func, &[SimValue::I64(10), SimValue::I64(3)]);
    assert!(r1.error.is_none(), "{:?}", r1.error);
    assert_eq!(r1.returns, vec![SimValue::I64(10)]);

    let mut sim = Simulator::new(0);
    let r2 = sim.run(&func, &[SimValue::I64(3), SimValue::I64(10)]);
    assert!(r2.error.is_none(), "{:?}", r2.error);
    assert_eq!(r2.returns, vec![SimValue::I64(10)]);
}

// 5. Sum 0..N via back-edge brif (loop header with block params).
#[test]
fn loop_sum() {
    let mut func = empty_func(&[types::I64], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        let header = bcx.create_block();
        let exit = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.append_block_param(header, types::I64); // i
        bcx.append_block_param(header, types::I64); // acc
        bcx.append_block_param(exit, types::I64);

        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let n = bcx.block_params(entry)[0];
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.ins().jump(header, &[zero.into(), zero.into()]);

        bcx.switch_to_block(header);
        let i = bcx.block_params(header)[0];
        let acc = bcx.block_params(header)[1];
        let cmp = bcx.ins().icmp(IntCC::SignedLessThan, i, n);
        let next_acc = bcx.ins().iadd(acc, i);
        let next_i = bcx.ins().iadd_imm(i, 1);
        bcx.ins().brif(
            cmp,
            header,
            &[next_i.into(), next_acc.into()],
            exit,
            &[acc.into()],
        );
        bcx.seal_block(header);
        bcx.seal_block(exit);

        bcx.switch_to_block(exit);
        let r = bcx.block_params(exit)[0];
        bcx.ins().return_(&[r]);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    let result = sim.run(&func, &[SimValue::I64(10)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    // 0 + 1 + ... + 9 = 45
    assert_eq!(result.returns, vec![SimValue::I64(45)]);
}

// 6. Stubbed call: declared FuncRef, two i64 args. Verify stub returned a
// zero result and recorded the callee in the trace.
#[test]
fn call_stub() {
    let mut func = empty_func(&[types::I64, types::I64], &[types::I64]);

    // Declare an imported "extern_add" with i64,i64 -> i64.
    let mut callee_sig = Signature::new(CallConv::SystemV);
    callee_sig.params.push(AbiParam::new(types::I64));
    callee_sig.params.push(AbiParam::new(types::I64));
    callee_sig.returns.push(AbiParam::new(types::I64));
    let sig_ref = func.import_signature(callee_sig);
    let name_ref = func.declare_imported_user_function(UserExternalName::new(0, 1));
    let func_ref = func.import_function(ExtFuncData {
        name: ExternalName::user(name_ref),
        signature: sig_ref,
        colocated: false,
        patchable: false,
    });

    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        let b = bcx.block_params(entry)[1];
        let call = bcx.ins().call(func_ref, &[a, b]);
        let rs = bcx.inst_results(call).to_vec();
        bcx.ins().return_(&rs);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    sim.trace = true;
    let result = sim.run(&func, &[SimValue::I64(7), SimValue::I64(35)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I64(0)]);
    let trace_blob = result.trace.join("\n");
    assert!(
        trace_blob.contains("call "),
        "expected call line in trace, got:\n{trace_blob}"
    );
    assert!(
        trace_blob.contains("I64(7)") && trace_blob.contains("I64(35)"),
        "expected arg values in trace, got:\n{trace_blob}"
    );
}

// 6b. Stubbed call with a `&str` arg: exercises `jit_call!`'s `&'static str`
// lowering (ptr + len fat pointer = two pointer-sized iconsts) against the
// sim's call stub. Verifies the call landed, the stub returned zero, and
// the lowered length value appears in the trace.
#[test]
fn call_with_str() {
    let mut func = empty_func(&[], &[types::I64]);

    // Imported callee: fn(&str) -> i64, i.e. (data_ptr, len) -> i64 on a
    // 64-bit target.
    let mut callee_sig = Signature::new(CallConv::SystemV);
    callee_sig.params.push(AbiParam::new(types::I64));
    callee_sig.params.push(AbiParam::new(types::I64));
    callee_sig.returns.push(AbiParam::new(types::I64));
    let sig_ref = func.import_signature(callee_sig);
    let name_ref = func.declare_imported_user_function(UserExternalName::new(0, 1));
    let func_ref = func.import_function(ExtFuncData {
        name: ExternalName::user(name_ref),
        signature: sig_ref,
        colocated: false,
        patchable: false,
    });

    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        // `jit_call!` lowers `"hello, world!"` into two iconsts (ptr, len=13)
        // and emits the call. No Module needed — ptr_ty is supplied directly.
        let call = jit_call!(&mut bcx, types::I64, func_ref; "hello, world!");
        let rs = bcx.inst_results(call).to_vec();
        bcx.ins().return_(&rs);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    sim.trace = true;
    let result = sim.run(&func, &[]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I64(0)]);
    let trace_blob = result.trace.join("\n");
    assert!(
        trace_blob.contains("call "),
        "expected call line in trace, got:\n{trace_blob}"
    );
    assert!(
        trace_blob.contains("I64(13)"),
        "expected len=13 arg in trace, got:\n{trace_blob}"
    );
}

// 7. Dump format: sanity-check the section headers + a memory row.
#[test]
fn dump_sections() {
    let mut func = empty_func(&[], &[types::I64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let v = bcx.ins().iconst(types::I64, 0x4142_4344);
        bcx.ins().return_(&[v]);
        bcx.finalize();
    }

    let mut mem = vec![0u8; 32];
    mem[0..4].copy_from_slice(b"ABCD");
    let mut sim = Simulator::with_memory(mem);
    sim.trace = true;
    let result = sim.run(&func, &[]);
    assert!(result.error.is_none(), "{:?}", result.error);

    let mut buf: Vec<u8> = Vec::new();
    result.dump_to(&mut buf).unwrap();
    let out = String::from_utf8(buf).unwrap();
    assert!(out.contains("=== Trace ==="), "{out}");
    assert!(out.contains("=== Returns ==="), "{out}");
    assert!(out.contains("ret[0] = I64(1094861636)"), "{out}");
    assert!(out.contains("=== Registers (SSA) ==="), "{out}");
    assert!(out.contains("=== Memory (32 bytes"), "{out}");
    assert!(out.contains("|ABCD"), "{out}");
}

// 8. `bitcast` lands in InstructionData::LoadNoOffset (it carries
// MemFlags), not Unary — verify it's still routed to apply_unary.
#[test]
fn bitcast_int_to_float() {
    let mut func = empty_func(&[types::I64], &[types::F64]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        let f = bcx.ins().bitcast(types::F64, MemFlags::new(), a);
        bcx.ins().return_(&[f]);
        bcx.finalize();
    }

    let bits = std::f64::consts::PI.to_bits() as i64;
    let mut sim = Simulator::new(0);
    let result = sim.run(&func, &[SimValue::I64(bits)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::F64(std::f64::consts::PI)]);
}

// 9. `uextend` must zero-extend, not sign-extend: I8(0xFF) -> I32(255).
#[test]
fn uextend_zero_extends_narrow() {
    let mut func = empty_func(&[types::I8], &[types::I32]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        let w = bcx.ins().uextend(types::I32, a);
        bcx.ins().return_(&[w]);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    // I8(0xFF) is -1 signed, 255 unsigned. uextend must give 255.
    let result = sim.run(&func, &[SimValue::I8(-1)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I32(255)]);

    // Sanity: sextend on the same input gives -1 (0xFFFFFFFF).
    let mut func2 = empty_func(&[types::I8], &[types::I32]);
    let mut ctx2 = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func2, &mut ctx2);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        let w = bcx.ins().sextend(types::I32, a);
        bcx.ins().return_(&[w]);
        bcx.finalize();
    }
    let mut sim = Simulator::new(0);
    let result = sim.run(&func2, &[SimValue::I8(-1)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I32(-1)]);
}

// 10. Narrow unsigned arithmetic must mask to the operand width before
// the unsigned op: I8(200) udiv I8(2) -> I8(100), not the sign-extended
// `0xFFFF_FFFF_FFFF_FFC8 / 2`.
#[test]
fn udiv_narrow_unsigned() {
    let mut func = empty_func(&[types::I8, types::I8], &[types::I8]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        let b = bcx.block_params(entry)[1];
        let q = bcx.ins().udiv(a, b);
        bcx.ins().return_(&[q]);
        bcx.finalize();
    }

    // 200 unsigned is -56 signed in i8.
    let mut sim = Simulator::new(0);
    let result = sim.run(&func, &[SimValue::I8(-56), SimValue::I8(2)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    // 200 / 2 = 100, which is in i8 range.
    assert_eq!(result.returns, vec![SimValue::I8(100)]);
}

// 11. `icmp_imm` must wrap the immediate to the operand's width. With
// I8 operands, an immediate of 200 is i8::-56, so `0 < 200` (unsigned)
// is true but `0 SignedLessThan 200` should be false (since 200 wraps
// to -56).
#[test]
fn icmp_imm_wraps_immediate_to_operand_width() {
    let mut func = empty_func(&[types::I8], &[types::I8]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);
        let a = bcx.block_params(entry)[0];
        let cmp = bcx.ins().icmp_imm(IntCC::SignedLessThan, a, 200);
        bcx.ins().return_(&[cmp]);
        bcx.finalize();
    }

    // a = 0, imm = 200 wraps to -56 as i8.
    // SignedLessThan: 0 < -56 → false.
    let mut sim = Simulator::new(0);
    let result = sim.run(&func, &[SimValue::I8(0)]);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.returns, vec![SimValue::I8(0)]);
}

// 12. A function with a back-edge (a block that jumps to itself) would spin
// forever; `max_steps` must break it with `StepLimitExceeded`.
#[test]
fn infinite_loop_halts_on_step_limit() {
    let mut func = empty_func(&[], &[]);
    let mut ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut ctx);
        let entry = bcx.create_block();
        let loop_blk = bcx.create_block();
        bcx.switch_to_block(entry);
        bcx.ins().jump(loop_blk, &[]);
        bcx.seal_block(entry);
        bcx.switch_to_block(loop_blk);
        bcx.ins().jump(loop_blk, &[]); // back-edge to itself
        bcx.seal_block(loop_blk);
        bcx.finalize();
    }

    let mut sim = Simulator::new(0);
    sim.max_steps = 1_000; // keep the test fast
    let result = sim.run(&func, &[]);
    assert!(
        matches!(
            result.error,
            Some(SimError::StepLimitExceeded { limit: 1_000 })
        ),
        "expected StepLimitExceeded, got {:?}",
        result.error
    );
}
