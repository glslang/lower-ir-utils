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

use lower_ir_utils::sim::{SimValue, Simulator};

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
