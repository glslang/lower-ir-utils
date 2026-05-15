//! Debug helpers for inspecting JIT-compiled machine code.
//!
//! Behind the `disas` Cargo feature: pulls in [`capstone`] (the same
//! disassembler Cranelift itself uses behind its own `disas` feature) and
//! exposes two entry points:
//!
//! - [`format_disassembly`] — turns a raw byte slice into a side-by-side
//!   listing, one instruction per line, with hex bytes on the left and the
//!   mnemonic translation on the right (the format an `objdump -d` dump
//!   would produce).
//! - [`define_function_with_disasm`] — mirrors [`crate::define_function`]
//!   but additionally captures the compiled bytes and a formatted
//!   disassembly, returning both alongside the [`FuncId`].
//!
//! Capstone is configured from the target ISA's name, so the helper
//! transparently works on whichever host arch the JIT is targeting
//! (currently x86_64, aarch64, riscv64, s390x).
//!
//! # Example
//!
//! ```ignore
//! use lower_ir_utils::{define_function_with_disasm, jit_signature};
//! use cranelift_module::Linkage;
//!
//! let (id, dump) = define_function_with_disasm(
//!     &mut module,
//!     "wrap",
//!     Linkage::Export,
//!     jit_signature!(&module; fn(i64) -> i64),
//!     |bcx, _module, params| params[0],
//! )?;
//! println!("{}", dump.text);
//! // 0x0000  48 89 f8           mov rax, rdi
//! // 0x0003  c3                 ret
//! ```

use std::fmt::{self, Write as _};

use capstone::arch::{self, BuildsCapstone, BuildsCapstoneExtraMode, BuildsCapstoneSyntax};
use capstone::Capstone;
use cranelift_codegen::isa::TargetIsa;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{FuncId, Linkage, Module, ModuleError};

use crate::builder::{declare_and_build, IntoReturns};
use cranelift_codegen::ir::{Signature, Value};

/// Captured artifacts from a JIT compile: the raw machine code and a
/// side-by-side disassembly string.
#[derive(Debug, Clone)]
pub struct JitDisasm {
    /// Final machine code for the compiled function, copied out of
    /// Cranelift's `Context` before it was cleared.
    pub bytes: Vec<u8>,
    /// Side-by-side dump of `bytes` — one Capstone instruction per line,
    /// `0xADDR  hex bytes  mnemonic operands`. Same format
    /// [`format_disassembly`] produces.
    pub text: String,
}

/// Error returned by [`format_disassembly`].
#[derive(Debug)]
pub enum DisasmError {
    /// The target ISA isn't one we know how to drive Capstone for.
    /// Contains the [`TargetIsa::name`] string for context.
    UnsupportedArch(String),
    /// Capstone failed to build or decode the byte stream.
    Capstone(capstone::Error),
    /// `writeln!`-style formatting failure (vanishingly rare, but the
    /// `fmt::Write` API surfaces it).
    Fmt(fmt::Error),
}

impl fmt::Display for DisasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisasmError::UnsupportedArch(arch) => {
                write!(f, "disassembly not implemented for target ISA `{arch}`")
            }
            DisasmError::Capstone(err) => write!(f, "capstone error: {err}"),
            DisasmError::Fmt(err) => write!(f, "format error: {err}"),
        }
    }
}

impl std::error::Error for DisasmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DisasmError::UnsupportedArch(_) => None,
            DisasmError::Capstone(err) => Some(err),
            DisasmError::Fmt(err) => Some(err),
        }
    }
}

impl From<capstone::Error> for DisasmError {
    fn from(err: capstone::Error) -> Self {
        DisasmError::Capstone(err)
    }
}

impl From<fmt::Error> for DisasmError {
    fn from(err: fmt::Error) -> Self {
        DisasmError::Fmt(err)
    }
}

/// Error returned by [`define_function_with_disasm`].
#[derive(Debug)]
pub enum DefineFunctionWithDisasmError {
    /// Wrapped [`ModuleError`] from any of the underlying Cranelift module
    /// operations (declare / define / clear).
    Module(ModuleError),
    /// Disassembly formatting failed after the function compiled
    /// successfully. The compile is still in the module's state at this
    /// point.
    Disasm(DisasmError),
}

impl fmt::Display for DefineFunctionWithDisasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DefineFunctionWithDisasmError::Module(err) => write!(f, "{err}"),
            DefineFunctionWithDisasmError::Disasm(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for DefineFunctionWithDisasmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DefineFunctionWithDisasmError::Module(err) => Some(err),
            DefineFunctionWithDisasmError::Disasm(err) => Some(err),
        }
    }
}

impl From<ModuleError> for DefineFunctionWithDisasmError {
    fn from(err: ModuleError) -> Self {
        DefineFunctionWithDisasmError::Module(err)
    }
}

impl From<DisasmError> for DefineFunctionWithDisasmError {
    fn from(err: DisasmError) -> Self {
        DefineFunctionWithDisasmError::Disasm(err)
    }
}

/// Build a [`Capstone`] decoder for `isa`'s instruction set.
fn capstone_for(isa: &dyn TargetIsa) -> Result<Capstone, DisasmError> {
    let cs = match isa.name() {
        "x64" => Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .syntax(arch::x86::ArchSyntax::Intel)
            .detail(false)
            .build()?,
        "aarch64" => Capstone::new()
            .arm64()
            .mode(arch::arm64::ArchMode::Arm)
            .detail(false)
            .build()?,
        "riscv64" => Capstone::new()
            .riscv()
            .mode(arch::riscv::ArchMode::RiscV64)
            .extra_mode(std::iter::once(arch::riscv::ArchExtraMode::RiscVC))
            .detail(false)
            .build()?,
        "s390x" => Capstone::new()
            .sysz()
            .mode(arch::sysz::ArchMode::Default)
            .detail(false)
            .build()?,
        other => return Err(DisasmError::UnsupportedArch(other.to_string())),
    };
    Ok(cs)
}

/// Format raw machine code as a side-by-side opcode / mnemonic dump.
///
/// Each line is `0xADDR  hex bytes (padded)  mnemonic operands` — the
/// classic disassembler layout, with bytes on the left and assembly on the
/// right. Returns the full multi-line string.
///
/// The `isa` is only used to pick the right Capstone decoder; the byte
/// slice is treated as a flat instruction stream starting at offset 0.
pub fn format_disassembly(bytes: &[u8], isa: &dyn TargetIsa) -> Result<String, DisasmError> {
    let cs = capstone_for(isa)?;
    let insns = cs.disasm_all(bytes, 0)?;

    let mut out = String::new();
    for ins in insns.iter() {
        let hex: String = ins
            .bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        let mnem = ins.mnemonic().unwrap_or("");
        let ops = ins.op_str().unwrap_or("");
        let sep = if ops.is_empty() { "" } else { " " };
        // 24 chars covers the longest realistic Cranelift x86_64 emission
        // (worst-case fixed 8 bytes = 23 chars with spaces).
        writeln!(out, "0x{:04x}  {hex:<24}  {mnem}{sep}{ops}", ins.address())?;
    }
    Ok(out)
}

/// Define a JIT function exactly like [`crate::define_function`], but also
/// capture the compiled machine code and a side-by-side disassembly.
///
/// Internally this:
/// 1. Declares the function and builds the IR (identical to
///    [`crate::define_function`]).
/// 2. Enables disassembly capture on the compile context.
/// 3. Calls `module.define_function`, which compiles to machine code.
/// 4. Copies bytes out of `ctx.compiled_code()` before clearing the
///    context.
/// 5. Runs [`format_disassembly`] using `module.isa()`.
///
/// On success returns the [`FuncId`] alongside a [`JitDisasm`] holding the
/// raw bytes and the formatted text.
#[allow(clippy::result_large_err)]
pub fn define_function_with_disasm<M, F, R>(
    module: &mut M,
    name: &str,
    linkage: Linkage,
    signature: Signature,
    body: F,
) -> Result<(FuncId, JitDisasm), DefineFunctionWithDisasmError>
where
    M: Module,
    F: FnOnce(&mut FunctionBuilder<'_>, &mut M, &[Value]) -> R,
    R: IntoReturns,
{
    let (func_id, mut ctx) = declare_and_build(module, name, linkage, signature, body)?;
    ctx.set_disasm(true);
    module.define_function(func_id, &mut ctx)?;

    let bytes = ctx
        .compiled_code()
        .expect("compiled_code is populated immediately after define_function")
        .code_buffer()
        .to_vec();
    let text = format_disassembly(&bytes, module.isa())?;

    module.clear_context(&mut ctx);
    Ok((func_id, JitDisasm { bytes, text }))
}
