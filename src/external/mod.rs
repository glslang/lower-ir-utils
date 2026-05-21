//! Wrappers that implement [`JitParam`](crate::JitParam) /
//! [`JitArg`](crate::JitArg) for types defined in other crates.
//!
//! Rust's orphan rule prevents downstream code from implementing these traits
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
