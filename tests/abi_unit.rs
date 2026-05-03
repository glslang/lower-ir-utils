//! Unit tests for the type-level [`JitParam`] mapping.
//!
//! These don't need a full JIT — we just call `push_params` and inspect the
//! resulting `AbiParam` vector.

use cranelift_codegen::ir::{types, AbiParam, Type};
use lower_ir_utils::JitParam;

const PTR64: Type = types::I64;

fn collect<T: JitParam>() -> Vec<AbiParam> {
    let mut v = Vec::new();
    T::push_params(&mut v, PTR64);
    v
}

fn types_of(params: &[AbiParam]) -> Vec<Type> {
    params.iter().map(|p| p.value_type).collect()
}

#[test]
fn unit_pushes_nothing() {
    assert!(collect::<()>().is_empty());
}

#[test]
fn integer_widths() {
    assert_eq!(types_of(&collect::<i8>()), vec![types::I8]);
    assert_eq!(types_of(&collect::<u8>()), vec![types::I8]);
    assert_eq!(types_of(&collect::<i16>()), vec![types::I16]);
    assert_eq!(types_of(&collect::<u16>()), vec![types::I16]);
    assert_eq!(types_of(&collect::<i32>()), vec![types::I32]);
    assert_eq!(types_of(&collect::<u32>()), vec![types::I32]);
    assert_eq!(types_of(&collect::<i64>()), vec![types::I64]);
    assert_eq!(types_of(&collect::<u64>()), vec![types::I64]);
    assert_eq!(types_of(&collect::<bool>()), vec![types::I8]);
}

#[test]
fn floats() {
    assert_eq!(types_of(&collect::<f32>()), vec![types::F32]);
    assert_eq!(types_of(&collect::<f64>()), vec![types::F64]);
}

#[test]
fn usize_isize_use_pointer_type() {
    assert_eq!(types_of(&collect::<usize>()), vec![PTR64]);
    assert_eq!(types_of(&collect::<isize>()), vec![PTR64]);
}

#[test]
fn raw_pointers_are_one_pointer() {
    assert_eq!(types_of(&collect::<*const i32>()), vec![PTR64]);
    assert_eq!(types_of(&collect::<*mut u8>()), vec![PTR64]);
    assert_eq!(types_of(&collect::<*const [u8; 16]>()), vec![PTR64]);
}

#[test]
fn references_to_sized_are_one_pointer() {
    assert_eq!(types_of(&collect::<&i32>()), vec![PTR64]);
    assert_eq!(types_of(&collect::<&mut u64>()), vec![PTR64]);
}

#[test]
fn str_is_two_pointers() {
    assert_eq!(types_of(&collect::<&str>()), vec![PTR64, PTR64]);
}

#[test]
fn slice_is_two_pointers() {
    assert_eq!(types_of(&collect::<&[u8]>()), vec![PTR64, PTR64]);
    assert_eq!(types_of(&collect::<&mut [i32]>()), vec![PTR64, PTR64]);
}

#[test]
fn ptr_ty_is_threaded_through() {
    let mut v = Vec::new();
    <usize as JitParam>::push_params(&mut v, types::I32);
    assert_eq!(types_of(&v), vec![types::I32]);
}
