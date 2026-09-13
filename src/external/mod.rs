//! Wrappers that implement [`JitArg`](crate::JitArg) constant lowering
//! for types defined in other crates. These are not native signature types.
//!
//! Rust's orphan rule prevents downstream code from implementing this trait
//! directly on foreign types, so each supported upstream crate gets a
//! submodule of newtype wrappers here. Every submodule is gated behind a
//! Cargo feature of the same name — enabling none of them keeps the
//! dependency list lean.
//!
//! Currently:
//!
//! - [`chrono`]: wrappers for `chrono::NaiveDate`, `NaiveTime`, and
//!   `NaiveDateTime` (feature `chrono`).

#[cfg(feature = "chrono")]
pub mod chrono;
