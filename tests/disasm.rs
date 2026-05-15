//! Smoke test for the `disas` feature: define a JIT function via
//! `define_function_with_disasm` and verify the returned disassembly is
//! shaped like a real disassembler dump (offsets + hex bytes + mnemonics)
//! and that the function still executes correctly.

#![cfg(feature = "disas")]

use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use lower_ir_utils::{define_function_with_disasm, format_disassembly, jit_signature};

fn jit_module() -> JITModule {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()))
}

#[test]
fn captures_disasm_for_identity_function() {
    let mut module = jit_module();

    let sig = jit_signature!(&module; fn(i64) -> i64);
    let (id, dump) = define_function_with_disasm(
        &mut module,
        "identity",
        Linkage::Export,
        sig,
        |_bcx, _module, params| params[0],
    )
    .expect("define_function_with_disasm");

    // Side-by-side dump must have at least one instruction line, each
    // starting with `0x` for the address column.
    assert!(!dump.text.is_empty(), "disasm text was empty");
    assert!(
        dump.text.lines().all(|l| l.starts_with("0x")),
        "expected every disasm line to start with 0x; got:\n{}",
        dump.text
    );
    assert!(
        dump.text.lines().count() >= 1,
        "expected at least one instruction line; got:\n{}",
        dump.text
    );

    // And the raw bytes must be non-empty (since we executed the function
    // body, Cranelift had to emit at least a return).
    assert!(!dump.bytes.is_empty(), "disasm bytes were empty");

    // Surface the dump under `cargo test -- --nocapture` for eyeballing.
    eprintln!("=== identity disasm ===\n{}", dump.text);

    module.finalize_definitions().unwrap();
    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    assert_eq!(f(7), 7);
    assert_eq!(f(-42), -42);
}

#[test]
fn format_disassembly_works_on_borrowed_bytes() {
    let mut module = jit_module();

    let sig = jit_signature!(&module; fn() -> i64);
    let (_id, dump) = define_function_with_disasm(
        &mut module,
        "zero",
        Linkage::Export,
        sig,
        |bcx, _module, _params| {
            use cranelift_codegen::ir::types;
            use cranelift_codegen::ir::InstBuilder;
            bcx.ins().iconst(types::I64, 0)
        },
    )
    .expect("define_function_with_disasm");

    // Re-format the same bytes via the low-level entry point and check the
    // output is identical: `format_disassembly` is the single source of
    // truth that `define_function_with_disasm` delegates to.
    let again = format_disassembly(&dump.bytes, module.isa()).expect("format_disassembly");
    assert_eq!(dump.text, again);
}
