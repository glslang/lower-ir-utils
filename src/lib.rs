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
//! # Using in an async runtime (tokio)
//!
//! Cranelift compilation — chiefly [`JITModule::finalize_definitions`] and the
//! per-function `module.define_function` driven by [`define_function`] /
//! [`define_jit_fn!`] — is a synchronous, CPU-bound, blocking step. Calling it
//! directly on a tokio worker thread stalls the executor. Keep it off the async
//! workers:
//!
//! - **Compile off-thread.** With the `tokio` feature, [`spawn_blocking_build`]
//!   moves the module onto [`tokio::task::spawn_blocking`] and returns it so you
//!   can pull function pointers out afterward. (Or call `spawn_blocking`
//!   yourself.)
//! - **Finalized function pointers are runtime-agnostic.** They are plain
//!   `extern "C" fn` — `Copy` and `Send` — so once obtained they can be stored
//!   and called from any task on any worker; awaiting between obtaining and
//!   calling is fine.
//! - **The module owns the executable memory.** It must outlive every call to
//!   any function pointer obtained from it; keep it alive (in scope, an `Arc`,
//!   or task-local state) for as long as the JITed code runs.
//! - **Embedded immediates must stay valid.** [`JitArg`] for `&'static T`,
//!   `&str`, `&[T]`, or `*const T` bakes a host address into the IR. Across
//!   `.await` this is only sound when the data is genuinely `'static`; never
//!   embed a pointer to data that may be dropped while a task is suspended.
//!
//! ```ignore
//! let mut module = JITModule::new(jb); // imports registered on `jb` first
//! let ext_id = double_i64_jit::declare(&mut module);
//!
//! let (module, wrap_id) = lower_ir_utils::spawn_blocking_build(module, move |m| {
//!     let id = define_jit_fn!(m, "wrap", Linkage::Export, fn(i64) -> i64,
//!         |bcx, m, p| double_i64_jit::call(bcx, m, ext_id, p[0])).unwrap();
//!     m.finalize_definitions().unwrap();
//!     id
//! })
//! .await;
//!
//! let f: extern "C" fn(i64) -> i64 =
//!     unsafe { std::mem::transmute(module.get_finalized_function(wrap_id)) };
//! assert_eq!(f(21), 42);
//! // `module` must stay alive while `f` is callable.
//! ```
//!
//! [`JITModule::finalize_definitions`]: cranelift_jit::JITModule::finalize_definitions
//!
//! # Main items
//!
//! - Traits: **[`JitParam`]** (Rust type → [`AbiParam`](cranelift_codegen::ir::AbiParam)s), **[`JitArg`]**
//!   (Rust value → [`Value`](cranelift_codegen::ir::Value) results via [`InstBuilder`](cranelift_codegen::ir::InstBuilder)).
//! - Macros: **`jit_signature!`**, **`jit_call!`**, **`define_jit_fn!`** (exported at
//!   the crate root).
//! - **`define_function`**, **[`IntoReturns`]**: declare and define a function in one step.
//! - Attribute macro: **`jit_export`** (re-export from `lower_ir_utils_macros`). The
//!   generated `<fn>_jit` module includes **`try_declare`** (fallible) alongside **`declare`**.
//! - Tuple returns from **`#[jit_export]`**: `<fn>_jit::call` returns
//!   [`Inst`](cranelift_codegen::ir::Inst); use `bcx.inst_results(inst)` — see the README
//!   "Tuple returns" section.
//!
//! # Optional Cargo features
//!
//! All optional features are **off by default** (`docs.rs` builds with `all-features = true`).
//!
//! - **`disas`** — `disasm` module (`define_function_with_disasm`, `format_disassembly`):
//!   Capstone side-by-side opcode dumps.
//! - **`sim`** — `sim` module (`Simulator`, `SimValue`, `SimResult`): IR interpreter over a
//!   flat byte buffer (debug aid; host `call`s are stubbed).
//! - **`chrono`** — `external::chrono` wrappers (`JitNaiveDate`, `JitNaiveTime`,
//!   `JitNaiveDateTime`) for naive `chrono` date/time types.
//! - **`tokio`** — `runtime` module (`spawn_blocking_build`): moves the blocking
//!   Cranelift compile/finalize step onto tokio's blocking thread pool.
//!
//! The crate README (also on docs.rs) adds runnable sketches and links to integration tests.

pub mod abi;
pub mod builder;
#[cfg(feature = "disas")]
pub mod disasm;
#[cfg(feature = "chrono")]
pub mod external;
mod macros;
#[cfg(feature = "tokio")]
pub mod runtime;
#[cfg(feature = "sim")]
pub mod sim;

pub use abi::{JitArg, JitParam};
pub use builder::{define_function, IntoReturns};
#[cfg(feature = "disas")]
pub use disasm::{
    define_function_with_disasm, format_disassembly, DefineFunctionWithDisasmError, DisasmError,
    JitDisasm,
};
#[cfg(feature = "chrono")]
pub use external::chrono::{JitNaiveDate, JitNaiveDateTime, JitNaiveTime};
pub use lower_ir_utils_macros::jit_export;
#[cfg(feature = "tokio")]
pub use runtime::spawn_blocking_build;

#[doc(hidden)]
pub mod __reexport {
    pub use cranelift_codegen;
    pub use cranelift_frontend;
    pub use cranelift_jit;
    pub use cranelift_module;
    pub use smallvec;
}
