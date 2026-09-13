//! External consumer of `lower-ir-utils`.
//!
//! This crate's only `[dependencies]` entry is `lower-ir-utils`. It exists to
//! prove that the `#[jit_export]` proc-macro's generated code resolves all
//! cranelift paths via `lower_ir_utils::__reexport::*`, and not via the user
//! crate's own deps. If the macro ever regresses to absolute paths like
//! `::cranelift_jit::JITBuilder`, this crate will fail to compile.

//! Unsupported types are also rejected without direct Cranelift dependencies:
//!
//! ```compile_fail,E0277
//! type Pair = (i32, i32);
//! #[lower_ir_utils::jit_export]
//! fn invalid() -> Pair { (1, 2) }
//! ```
//!
//! ```compile_fail,E0277
//! type Text = &'static str;
//! fn build(module: &lower_ir_utils::__reexport::cranelift_jit::JITModule) {
//!     let _ = lower_ir_utils::jit_signature!(module; fn(Text));
//! }
//! ```
//!
//! ```compile_fail,E0277
//! fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
//!     let _ = lower_ir_utils::define_jit_fn!(
//!         module, "invalid", lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
//!         fn() -> (i64, f64, i64), |_, _, _| (),
//!     );
//! }
//! ```

use lower_ir_utils::jit_export;

#[jit_export]
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[jit_export]
pub fn lookup_len(_data: *const u8, len: usize) -> i64 {
    len as i64
}

#[jit_export]
pub fn record(_x: i64) {}
