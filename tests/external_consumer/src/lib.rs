//! External consumer of `lower-ir-utils`.
//!
//! This crate's only `[dependencies]` entry is `lower-ir-utils`. It exists to
//! prove that the `#[jit_export]` proc-macro's generated code resolves all
//! cranelift paths via `lower_ir_utils::__reexport::*`, and not via the user
//! crate's own deps. If the macro ever regresses to absolute paths like
//! `::cranelift_jit::JITBuilder`, this crate will fail to compile.

use lower_ir_utils::jit_export;

#[jit_export]
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[jit_export]
pub fn lookup_len(s: &str) -> i64 {
    s.len() as i64
}

#[jit_export]
pub fn record(_x: i64) {}
