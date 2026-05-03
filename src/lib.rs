//! Macros and traits for bridging Rust types to Cranelift JIT signatures and call sites.
//!
//! See [`JitParam`] (type-level signature shape) and [`JitArg`] (value-level lowering),
//! plus the [`jit_signature!`] and [`jit_call!`] macros that compose them.

pub mod abi;
pub mod builder;
mod macros;

pub use abi::{JitArg, JitParam};
pub use builder::{define_function, IntoReturns};
pub use lower_ir_utils_macros::jit_export;

#[doc(hidden)]
pub mod __reexport {
    pub use cranelift_codegen;
    pub use cranelift_module;
    pub use smallvec;
}
