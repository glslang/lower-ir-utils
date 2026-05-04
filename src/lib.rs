//! Bridges Rust types to [Cranelift](https://cranelift.dev/)
//! [`Signature`](cranelift_codegen::ir::Signature) values, lowers values at JIT `call`
//! sites, and trims [`Module`](cranelift_module::Module) /
//! [`FunctionBuilder`](cranelift_frontend::FunctionBuilder) boilerplate while leaving the
//! underlying APIs in your hands.
//!
//! **Cranelift / crate versions.** This crate targets **Cranelift 0.131** (see
//! dependencies in `Cargo.toml`). Use matching `cranelift-*` versions in your
//! project to avoid subtle ABI or API skew.
//!
//! **`#[jit_export]`** is implemented in the companion crate
//! [lower-ir-utils-macros](https://docs.rs/lower-ir-utils-macros) and re-exported
//! here; see that crate's docs for details on the generated `<fn>_jit` module.
//!
//! # Platform and ABI notes
//!
//! [`JitParam`] / [`JitArg`] model `&str` and slices as **two machine words**
//! (data pointer and length), matching how separate `(ptr, len)` arguments look in
//! Cranelift. That matches common 64-bit C ABIs (e.g. separate scalar args).
//! **`#[jit_export]`** injects `extern "C"` when none is specified and allows
//! `improper_ctypes_definitions` so you can write `&str` in Rust signatures on targets
//! that pass fat pointers compatibly with that layout—on platforms where that does
//! not hold, flatten parameters to scalars explicitly.
//!
//! # Example (end-to-end JIT)
//!
//! Flow (register host symbol → declare import → wrap with [`define_jit_fn!`] →
//! finalize → call), adapted from `tests/external_consumer/tests/smoke.rs` in this
//! repository. Add `cranelift-jit`, `cranelift-module`, `cranelift-codegen`, and
//! `cranelift-native` alongside `lower-ir-utils` for this shape; or import the
//! Cranelift crates through [`__reexport`] and keep only `lower-ir-utils` as a normal
//! dependency, as that test crate does.
//!
//! ```ignore
//! use cranelift_jit::{JITBuilder, JITModule};
//! use cranelift_module::{default_libcall_names, Linkage};
//! use cranelift_codegen::settings::{self, Configurable};
//! use lower_ir_utils::{define_jit_fn, jit_export};
//!
//! #[jit_export]
//! fn add(a: i64, b: i64) -> i64 {
//!     a + b
//! }
//!
//! let mut flag_builder = settings::builder();
//! flag_builder.set("use_colocated_libcalls", "false").unwrap();
//! flag_builder.set("is_pic", "false").unwrap();
//! let isa = cranelift_native::builder()
//!     .unwrap()
//!     .finish(settings::Flags::new(flag_builder))
//!     .unwrap();
//!
//! let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
//! add_jit::register(&mut jb);
//! let mut module = JITModule::new(jb);
//! let ext_id = add_jit::declare(&mut module);
//!
//! let wrap_id = define_jit_fn!(
//!     &mut module, "wrap", Linkage::Export, fn(i64, i64) -> i64,
//!     |bcx, module, params| add_jit::call(bcx, module, ext_id, params[0], params[1]),
//! )
//! .unwrap();
//!
//! module.finalize_definitions().unwrap();
//! let f: extern "C" fn(i64, i64) -> i64 =
//!     unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
//! assert_eq!(f(2, 3), 5);
//! ```
//!
//! # Main items
//!
//! - Traits: **[`JitParam`]** (Rust type → [`AbiParam`](cranelift_codegen::ir::AbiParam)s), **[`JitArg`]**
//!   (Rust value → [`Value`](cranelift_codegen::ir::Value) results via [`InstBuilder`](cranelift_codegen::ir::InstBuilder)).
//! - Macros: **`jit_signature!`**, **`jit_call!`**, **`define_jit_fn!`** (exported at
//!   the crate root).
//! - **`define_function`**, **[`IntoReturns`]**: declare and define a function in one step.
//! - Attribute macro: **`jit_export`** (re-export from `lower_ir_utils_macros`).
//!
//! The crate README (also on docs.rs) adds another runnable sketch and links to more
//! integration tests.

pub mod abi;
pub mod builder;
mod macros;

pub use abi::{JitArg, JitParam};
pub use builder::{define_function, IntoReturns};
pub use lower_ir_utils_macros::jit_export;

#[doc(hidden)]
pub mod __reexport {
    pub use cranelift_codegen;
    pub use cranelift_frontend;
    pub use cranelift_jit;
    pub use cranelift_module;
    pub use smallvec;
}
