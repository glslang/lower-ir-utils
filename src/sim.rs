//! Quick-debug interpreter for finalized Cranelift IR.
//!
//! Behind the `sim` Cargo feature. Walks a
//! [`cranelift_codegen::ir::Function`] block by block, evaluating each
//! instruction against a flat `Vec<u8>` memory and an SSA register table,
//! then renders both side by side with the final return values. Useful for
//! the "did my codegen emit the right loads?" debugging loop — *not* a
//! replacement for a real debugger.
//!
//! # What's supported
//!
//! Integer + float scalar arithmetic, immediate-form arithmetic, bitwise +
//! shifts, `icmp`/`fcmp`, `select`, `load`/`store` (honoring the immediate
//! offset; `MemFlags` are recorded but not enforced), `brif`/`jump`/`return`,
//! and `call` (stubbed — see below). Casts (`bitcast`, `uextend`, `sextend`,
//! `ireduce`) round out the set so common pointer/integer mixing works.
//!
//! Anything not in that set produces a `SimError::UnsupportedOpcode` and
//! halts execution; the partial [`SimResult`] is still returned so
//! [`SimResult::dump`] can show how far the run got.
//!
//! # Host calls are stubbed
//!
//! Cranelift `call` instructions reference a `FuncRef` whose target is
//! generally an external symbol registered with the JIT. The simulator
//! cannot dispatch to host functions safely (it has no FFI trampoline), so
//! every `call` instead:
//!
//! 1. Logs the callee name (from `dfg.ext_funcs[func_ref].name`) and every
//!    argument value into the trace.
//! 2. Heuristically previews fat-pointer arg pairs (`i64` ptr + plausible
//!    `i64` length pointing into memory) as a UTF-8 string — purely
//!    cosmetic, for spotting `&str` mistakes.
//! 3. Returns zero-valued results matching the signature's return types,
//!    binding them to the call's result `Value`s.
//!
//! That's enough to verify the IR around the call is correct (right pointer,
//! right length, right arg order) without leaving the simulator.
//!
//! # Memory model
//!
//! `Simulator` owns a `Vec<u8>` flat heap. Pointer values are interpreted
//! as byte indices into that buffer; `load`/`store` honor the inst's
//! `Offset32` immediate. Reads/writes use host-endian byte order — that's
//! what Cranelift's default x86_64 emission would produce too, and good
//! enough for a debug tool. Out-of-bounds accesses produce a
//! `SimError::OutOfBounds`.
//!
//! # Example
//!
//! ```ignore
//! use cranelift_codegen::ir::Function;
//! use lower_ir_utils::sim::{Simulator, SimValue};
//!
//! let func: Function = /* build via FunctionBuilder, finalize */;
//! let mut sim = Simulator::with_memory(vec![0; 256]);
//! sim.trace = true;
//! let result = sim.run(&func, &[SimValue::I64(10), SimValue::I64(32)]);
//! result.dump();
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Write};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    types, Block, ExternalName, Function, InstructionData, Opcode, Type, Value,
};

/// A concrete value bound to a Cranelift SSA `Value` during simulation.
///
/// Mirrors the scalar subset of Cranelift types that `lower-ir-utils`
/// covers via [`crate::JitParam`]. Pointers ride as `I64` (the simulator
/// uses 64-bit indices into its flat memory buffer regardless of host
/// pointer width — `lower-ir-utils` is x86_64-leaning).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl SimValue {
    /// Coerce to a `u64` for use as an address or for bitwise reasoning.
    /// Integer values sign-extend; floats round-trip through `to_bits`.
    pub fn as_u64(self) -> u64 {
        match self {
            SimValue::I8(x) => x as i64 as u64,
            SimValue::I16(x) => x as i64 as u64,
            SimValue::I32(x) => x as i64 as u64,
            SimValue::I64(x) => x as u64,
            SimValue::F32(x) => x.to_bits() as u64,
            SimValue::F64(x) => x.to_bits(),
        }
    }

    /// Coerce to a signed 64-bit integer (sign-extended for narrower ints).
    pub fn as_i64(self) -> i64 {
        match self {
            SimValue::I8(x) => x as i64,
            SimValue::I16(x) => x as i64,
            SimValue::I32(x) => x as i64,
            SimValue::I64(x) => x,
            SimValue::F32(x) => x.to_bits() as i64,
            SimValue::F64(x) => x.to_bits() as i64,
        }
    }

    /// Zero of the given Cranelift type. Used for stubbed-call return
    /// values and uninitialized block params at the entry seam.
    pub fn zero_of(ty: Type) -> Option<Self> {
        Some(match ty {
            types::I8 => SimValue::I8(0),
            types::I16 => SimValue::I16(0),
            types::I32 => SimValue::I32(0),
            types::I64 => SimValue::I64(0),
            types::F32 => SimValue::F32(0.0),
            types::F64 => SimValue::F64(0.0),
            _ => return None,
        })
    }

    /// The Cranelift `Type` this value would inhabit.
    pub fn ty(self) -> Type {
        match self {
            SimValue::I8(_) => types::I8,
            SimValue::I16(_) => types::I16,
            SimValue::I32(_) => types::I32,
            SimValue::I64(_) => types::I64,
            SimValue::F32(_) => types::F32,
            SimValue::F64(_) => types::F64,
        }
    }
}

impl fmt::Display for SimValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimValue::I8(x) => write!(f, "I8({x})"),
            SimValue::I16(x) => write!(f, "I16({x})"),
            SimValue::I32(x) => write!(f, "I32({x})"),
            SimValue::I64(x) => write!(f, "I64({x})"),
            SimValue::F32(x) => write!(f, "F32({x})"),
            SimValue::F64(x) => write!(f, "F64({x})"),
        }
    }
}

/// Errors the simulator can surface mid-run. The partial [`SimResult`]
/// returned alongside still contains every value written before the fault.
#[derive(Debug, Clone)]
pub enum SimError {
    /// Opcode is outside the scalar/expression-lang subset the simulator
    /// covers. Carries the textual opcode name for the trace.
    UnsupportedOpcode(String),
    /// Load or store fell outside the memory buffer.
    OutOfBounds {
        addr: u64,
        size: usize,
        memory_len: usize,
    },
    /// A `Value` was referenced before any instruction wrote to it.
    /// Usually means a code path forgot to pass a block argument.
    UndefinedValue(Value),
    /// A computed type doesn't match what the instruction expects (e.g.
    /// `iadd` on an `F32`).
    TypeMismatch(String),
    /// Execution ran past [`Simulator::max_steps`] instructions without
    /// reaching a `return` — almost always a function with a back-edge
    /// (a loop), which this interpreter has no other way to break out of.
    StepLimitExceeded { limit: usize },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::UnsupportedOpcode(name) => write!(f, "unsupported opcode `{name}`"),
            SimError::OutOfBounds {
                addr,
                size,
                memory_len,
            } => write!(
                f,
                "out-of-bounds access: addr={addr:#x} size={size} memory_len={memory_len}"
            ),
            SimError::UndefinedValue(v) => write!(f, "undefined SSA value {v}"),
            SimError::TypeMismatch(s) => write!(f, "type mismatch: {s}"),
            SimError::StepLimitExceeded { limit } => {
                write!(
                    f,
                    "step limit exceeded ({limit} instructions) — possible infinite loop"
                )
            }
        }
    }
}

impl std::error::Error for SimError {}

/// Output of one simulation run.
///
/// Always populated even on error — fields hold whatever was written up to
/// the faulting instruction, so [`SimResult::dump`] still shows useful
/// state.
#[derive(Debug, Clone)]
pub struct SimResult {
    /// Values returned by the `return` instruction (empty if the run did
    /// not reach a return).
    pub returns: Vec<SimValue>,
    /// SSA register file at the end of the run. Ordered by `Value` for
    /// stable dumps.
    pub registers: BTreeMap<Value, SimValue>,
    /// Memory contents after the run.
    pub memory: Vec<u8>,
    /// Per-instruction trace (only populated when [`Simulator::trace`] was
    /// `true`).
    pub trace: Vec<String>,
    /// `Some(err)` if execution halted before a `return`.
    pub error: Option<SimError>,
}

/// Default instruction budget for [`Simulator::run`], used by
/// [`Simulator::new`] and [`Simulator::with_memory`]. Generous enough that no
/// realistic straight-line debug function comes close, while still bounding a
/// runaway loop to a fraction of a second.
pub const DEFAULT_MAX_STEPS: usize = 1_000_000;

/// The interpreter.
///
/// Build one with [`Simulator::new`] (zero-filled memory of the given size)
/// or [`Simulator::with_memory`] (caller-supplied initial bytes), then
/// call [`Simulator::run`] for each function you want to evaluate.
#[derive(Debug, Clone)]
pub struct Simulator {
    memory: Vec<u8>,
    /// If `true`, [`SimResult::trace`] is populated with one line per
    /// executed instruction. Defaults to `false`.
    pub trace: bool,
    /// Maximum number of instructions [`Simulator::run`] will execute before
    /// halting with [`SimError::StepLimitExceeded`]. Defaults to
    /// [`DEFAULT_MAX_STEPS`]. The functions this crate's `define_function`
    /// emits are straight-line, but hand-built or generated IR may contain a
    /// back-edge; this is the only thing that stops such a loop from spinning
    /// forever.
    pub max_steps: usize,
}

impl Simulator {
    /// Create a simulator with `size` bytes of zero-initialized memory.
    pub fn new(size: usize) -> Self {
        Self {
            memory: vec![0; size],
            trace: false,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    /// Create a simulator that takes ownership of a pre-populated byte
    /// buffer. Handy for testing `load` against known contents.
    pub fn with_memory(bytes: Vec<u8>) -> Self {
        Self {
            memory: bytes,
            trace: false,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    /// Read-only view of the simulator's memory.
    pub fn memory(&self) -> &[u8] {
        &self.memory
    }

    /// Evaluate `func` against `args`. The simulator's memory is moved
    /// into the returned [`SimResult`]; build a new [`Simulator`] for the
    /// next run if you need to preserve state across calls.
    pub fn run(&mut self, func: &Function, args: &[SimValue]) -> SimResult {
        let mut state = RunState {
            memory: std::mem::take(&mut self.memory),
            registers: BTreeMap::new(),
            trace: Vec::new(),
            record_trace: self.trace,
        };

        let entry = match func.layout.entry_block() {
            Some(b) => b,
            None => {
                return SimResult {
                    returns: Vec::new(),
                    registers: state.registers,
                    memory: state.memory,
                    trace: state.trace,
                    error: Some(SimError::UnsupportedOpcode("(no entry block)".into())),
                }
            }
        };

        // Bind entry-block params from caller args.
        for (param, arg) in func.dfg.block_params(entry).iter().zip(args.iter()) {
            state.registers.insert(*param, *arg);
        }

        let mut current = entry;
        let mut returns: Vec<SimValue> = Vec::new();
        let mut error: Option<SimError> = None;
        let mut steps: usize = 0;

        'outer: loop {
            // Snapshot the inst sequence — iterator borrows the layout,
            // and we need to keep `func` borrowed read-only.
            let insts: Vec<_> = func.layout.block_insts(current).collect();
            for inst in insts {
                // Bound total work so IR with a back-edge halts instead of
                // spinning forever (see `max_steps`).
                if steps >= self.max_steps {
                    let err = SimError::StepLimitExceeded {
                        limit: self.max_steps,
                    };
                    if state.record_trace {
                        state.trace.push(format!("  ! halt: {err} at {inst}"));
                    }
                    error = Some(err);
                    break 'outer;
                }
                steps += 1;

                let inst_data = &func.dfg.insts[inst];
                match Self::step(func, &mut state, inst, inst_data) {
                    StepResult::Continue => {}
                    StepResult::Jump(target, params) => {
                        for (slot, val) in func.dfg.block_params(target).iter().zip(params) {
                            state.registers.insert(*slot, val);
                        }
                        current = target;
                        continue 'outer;
                    }
                    StepResult::Return(vals) => {
                        returns = vals;
                        break 'outer;
                    }
                    StepResult::Error(err) => {
                        if state.record_trace {
                            state.trace.push(format!("  ! halt: {err} at {inst}"));
                        }
                        error = Some(err);
                        break 'outer;
                    }
                }
            }
            // Fell off the end of a block without a terminator — shouldn't
            // happen on a finalized function, but bail cleanly anyway.
            error = Some(SimError::UnsupportedOpcode(
                "(block has no terminator)".into(),
            ));
            break;
        }

        SimResult {
            returns,
            registers: state.registers,
            memory: state.memory,
            trace: state.trace,
            error,
        }
    }

    fn step(
        func: &Function,
        state: &mut RunState,
        inst: cranelift_codegen::ir::Inst,
        data: &InstructionData,
    ) -> StepResult {
        use StepResult::*;

        // ---- helpers closed over state.registers ----
        macro_rules! get {
            ($v:expr) => {
                match state.registers.get(&$v).copied() {
                    Some(x) => x,
                    None => return Error(SimError::UndefinedValue($v)),
                }
            };
        }

        let opcode = data.opcode();
        let result_ty = |i: usize| -> Option<Type> {
            func.dfg
                .inst_results(inst)
                .get(i)
                .map(|v| func.dfg.value_type(*v))
        };

        let mut produced: Vec<SimValue> = Vec::new();

        match data {
            // ----- constants -----
            InstructionData::UnaryImm { imm, .. } if opcode == Opcode::Iconst => {
                let raw: i64 = (*imm).into();
                let ty = result_ty(0).unwrap_or(types::I64);
                produced.push(match ty {
                    types::I8 => SimValue::I8(raw as i8),
                    types::I16 => SimValue::I16(raw as i16),
                    types::I32 => SimValue::I32(raw as i32),
                    types::I64 => SimValue::I64(raw),
                    _ => return Error(SimError::TypeMismatch(format!("iconst type {ty}"))),
                });
            }
            InstructionData::UnaryIeee32 { imm, .. } if opcode == Opcode::F32const => {
                produced.push(SimValue::F32(imm.as_f32()));
            }
            InstructionData::UnaryIeee64 { imm, .. } if opcode == Opcode::F64const => {
                produced.push(SimValue::F64(imm.as_f64()));
            }

            // ----- binary arith / bitwise -----
            InstructionData::Binary { args, .. } => {
                let a = get!(args[0]);
                let b = get!(args[1]);
                match apply_binary(opcode, a, b) {
                    Ok(v) => produced.push(v),
                    Err(e) => return Error(e),
                }
            }
            InstructionData::BinaryImm64 { arg, imm, .. } => {
                let a = get!(*arg);
                let imm_i64: i64 = (*imm).into();
                match apply_binary_imm(opcode, a, imm_i64) {
                    Ok(v) => produced.push(v),
                    Err(e) => return Error(e),
                }
            }

            // ----- unary arith / casts -----
            InstructionData::Unary { arg, .. } => {
                let a = get!(*arg);
                let ty = result_ty(0).unwrap_or_else(|| a.ty());
                match apply_unary(opcode, a, ty) {
                    Ok(v) => produced.push(v),
                    Err(e) => return Error(e),
                }
            }
            // `bitcast` is a load-shaped instruction in CLIF (it carries
            // `MemFlags`), so it lands here rather than `Unary`. Other
            // `LoadNoOffset`-format ops (e.g. atomic loads) are not
            // covered.
            InstructionData::LoadNoOffset { arg, .. } if opcode == Opcode::Bitcast => {
                let a = get!(*arg);
                let ty = result_ty(0).unwrap_or_else(|| a.ty());
                match apply_unary(opcode, a, ty) {
                    Ok(v) => produced.push(v),
                    Err(e) => return Error(e),
                }
            }

            // ----- compares -----
            InstructionData::IntCompare { args, cond, .. } => {
                let a = get!(args[0]);
                let b = get!(args[1]);
                produced.push(SimValue::I8(icmp(*cond, a, b) as i8));
            }
            InstructionData::IntCompareImm { arg, imm, cond, .. } => {
                let a = get!(*arg);
                let imm_i64: i64 = (*imm).into();
                // Wrap the immediate at the operand's width so signed and
                // unsigned compares both see the same value Cranelift would.
                let b = match a {
                    SimValue::I8(_) => SimValue::I8(imm_i64 as i8),
                    SimValue::I16(_) => SimValue::I16(imm_i64 as i16),
                    SimValue::I32(_) => SimValue::I32(imm_i64 as i32),
                    SimValue::I64(_) => SimValue::I64(imm_i64),
                    SimValue::F32(_) | SimValue::F64(_) => {
                        return Error(SimError::TypeMismatch("icmp_imm on float operand".into()));
                    }
                };
                produced.push(SimValue::I8(icmp(*cond, a, b) as i8));
            }
            InstructionData::FloatCompare { args, cond, .. } => {
                let a = get!(args[0]);
                let b = get!(args[1]);
                produced.push(SimValue::I8(fcmp(*cond, a, b) as i8));
            }

            // ----- ternary (select) -----
            InstructionData::Ternary { args, .. } if opcode == Opcode::Select => {
                let c = get!(args[0]);
                let x = get!(args[1]);
                let y = get!(args[2]);
                produced.push(if c.as_i64() != 0 { x } else { y });
            }

            // ----- memory -----
            InstructionData::Load { arg, offset, .. } => {
                let base = get!(*arg).as_u64();
                let off: i64 = (*offset).into();
                let addr = base.wrapping_add(off as u64) as usize;
                let ty = result_ty(0).unwrap_or(types::I64);
                match load_from(&state.memory, addr, ty) {
                    Ok(v) => produced.push(v),
                    Err(e) => return Error(e),
                }
            }
            InstructionData::Store { args, offset, .. } => {
                // ClIF: `store flags, x, p, offset` -> args[0]=x, args[1]=p
                let val = get!(args[0]);
                let base = get!(args[1]).as_u64();
                let off: i64 = (*offset).into();
                let addr = base.wrapping_add(off as u64) as usize;
                if let Err(e) = store_to(&mut state.memory, addr, val) {
                    return Error(e);
                }
            }

            // ----- control flow (terminators) -----
            InstructionData::Jump { destination, .. } => {
                let pool = &func.dfg.value_lists;
                let target = destination.block(pool);
                let mut params: Vec<SimValue> = Vec::new();
                for arg in destination.args(pool) {
                    if let Some(v) = arg.as_value() {
                        params.push(get!(v));
                    }
                }
                if state.record_trace {
                    state.trace.push(format!("  jump block{}", target.as_u32()));
                }
                return Jump(target, params);
            }
            InstructionData::Brif { arg, blocks, .. } => {
                let cond = get!(*arg).as_i64() != 0;
                let pool = &func.dfg.value_lists;
                let chosen = if cond { &blocks[0] } else { &blocks[1] };
                let target = chosen.block(pool);
                let mut params: Vec<SimValue> = Vec::new();
                for arg in chosen.args(pool) {
                    if let Some(v) = arg.as_value() {
                        params.push(get!(v));
                    }
                }
                if state.record_trace {
                    state.trace.push(format!(
                        "  brif {} -> block{} ({})",
                        cond,
                        target.as_u32(),
                        if cond { "then" } else { "else" }
                    ));
                }
                return Jump(target, params);
            }
            InstructionData::MultiAry { args, .. } if opcode == Opcode::Return => {
                let pool = &func.dfg.value_lists;
                let mut vals: Vec<SimValue> = Vec::new();
                for v in args.as_slice(pool) {
                    vals.push(get!(*v));
                }
                if state.record_trace {
                    let formatted: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                    state
                        .trace
                        .push(format!("  return [{}]", formatted.join(", ")));
                }
                return Return(vals);
            }

            // ----- call (stub) -----
            InstructionData::Call { args, func_ref, .. } => {
                let pool = &func.dfg.value_lists;
                let callee_data = &func.dfg.ext_funcs[*func_ref];
                let callee_name = format_extname(&callee_data.name, func);
                let mut arg_vals: Vec<SimValue> = Vec::with_capacity(args.len(pool));
                for v in args.as_slice(pool) {
                    arg_vals.push(get!(*v));
                }

                if state.record_trace {
                    state.trace.push(format!(
                        "  call {callee_name}({}){}",
                        format_call_args(&arg_vals, &state.memory),
                        format_call_results(func.dfg.inst_results(inst), &func.dfg),
                    ));
                }

                // Fill result Values with type-appropriate zeros.
                for v in func.dfg.inst_results(inst) {
                    let ty = func.dfg.value_type(*v);
                    let zero = match SimValue::zero_of(ty) {
                        Some(z) => z,
                        None => {
                            return Error(SimError::TypeMismatch(format!(
                                "call result type {ty} not supported"
                            )))
                        }
                    };
                    state.registers.insert(*v, zero);
                }
                return Continue;
            }

            _ => {
                return Error(SimError::UnsupportedOpcode(format!("{opcode}")));
            }
        }

        // Bind produced result(s) to inst result Value(s).
        let results = func.dfg.inst_results(inst);
        if state.record_trace {
            let lhs: Vec<String> = results.iter().map(|v| v.to_string()).collect();
            let rhs: Vec<String> = produced.iter().map(|v| v.to_string()).collect();
            let prefix = if lhs.is_empty() {
                String::new()
            } else {
                format!("{} = ", lhs.join(", "))
            };
            state
                .trace
                .push(format!("  {prefix}{opcode} [{}]", rhs.join(", ")));
        }
        for (slot, val) in results.iter().zip(produced) {
            state.registers.insert(*slot, val);
        }

        Continue
    }
}

// ---------- internal state ----------

struct RunState {
    memory: Vec<u8>,
    registers: BTreeMap<Value, SimValue>,
    trace: Vec<String>,
    record_trace: bool,
}

enum StepResult {
    Continue,
    Jump(Block, Vec<SimValue>),
    Return(Vec<SimValue>),
    Error(SimError),
}

// ---------- arithmetic helpers ----------

fn apply_binary(op: Opcode, a: SimValue, b: SimValue) -> Result<SimValue, SimError> {
    use SimValue::*;
    let mismatch = |what: &str| SimError::TypeMismatch(format!("{op}: {what}"));
    match (a, b) {
        (I8(x), I8(y)) => apply_int(op, x as i64, y as i64, 8).map(|r| I8(r as i8)),
        (I16(x), I16(y)) => apply_int(op, x as i64, y as i64, 16).map(|r| I16(r as i16)),
        (I32(x), I32(y)) => apply_int(op, x as i64, y as i64, 32).map(|r| I32(r as i32)),
        (I64(x), I64(y)) => apply_int(op, x, y, 64).map(I64),
        (F32(x), F32(y)) => apply_float(op, x as f64, y as f64).map(|r| F32(r as f32)),
        (F64(x), F64(y)) => apply_float(op, x, y).map(F64),
        _ => Err(mismatch("operand types differ")),
    }
}

fn apply_binary_imm(op: Opcode, a: SimValue, imm: i64) -> Result<SimValue, SimError> {
    use SimValue::*;
    let b = match a {
        I8(_) => I8(imm as i8),
        I16(_) => I16(imm as i16),
        I32(_) => I32(imm as i32),
        I64(_) => I64(imm),
        F32(_) | F64(_) => return Err(SimError::TypeMismatch(format!("{op}: imm on float"))),
    };
    // BinaryImm64 opcodes are integer-only. Map back through apply_binary.
    apply_binary(op, a, b)
}

/// Operates on the i64 sign-extended forms of the inputs. `width` is the
/// integer's width in bits — unsigned ops need to mask back to the
/// original width so that e.g. `udiv I8(200) I8(2)` doesn't see the
/// sign-extended `0xFFFF_FFFF_FFFF_FFC8`.
fn apply_int(op: Opcode, x: i64, y: i64, width: u32) -> Result<i64, SimError> {
    let mask: u64 = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let xu = (x as u64) & mask;
    let yu = (y as u64) & mask;
    let shamt = (y as u32) & (width - 1);
    Ok(match op {
        Opcode::Iadd | Opcode::IaddImm => x.wrapping_add(y),
        Opcode::Isub => x.wrapping_sub(y),
        Opcode::Imul | Opcode::ImulImm => x.wrapping_mul(y),
        Opcode::Sdiv | Opcode::SdivImm => x.checked_div(y).unwrap_or(0),
        Opcode::Udiv | Opcode::UdivImm => xu.checked_div(yu).unwrap_or(0) as i64,
        Opcode::Srem | Opcode::SremImm => x.checked_rem(y).unwrap_or(0),
        Opcode::Urem | Opcode::UremImm => xu.checked_rem(yu).unwrap_or(0) as i64,
        Opcode::Band | Opcode::BandImm => x & y,
        Opcode::Bor | Opcode::BorImm => x | y,
        Opcode::Bxor | Opcode::BxorImm => x ^ y,
        Opcode::Ishl | Opcode::IshlImm => x.wrapping_shl(shamt),
        Opcode::Ushr | Opcode::UshrImm => xu.wrapping_shr(shamt) as i64,
        Opcode::Sshr | Opcode::SshrImm => x.wrapping_shr(shamt),
        _ => return Err(SimError::UnsupportedOpcode(format!("{op}"))),
    })
}

fn apply_float(op: Opcode, x: f64, y: f64) -> Result<f64, SimError> {
    Ok(match op {
        Opcode::Fadd => x + y,
        Opcode::Fsub => x - y,
        Opcode::Fmul => x * y,
        Opcode::Fdiv => x / y,
        _ => return Err(SimError::UnsupportedOpcode(format!("{op}"))),
    })
}

fn apply_unary(op: Opcode, a: SimValue, result_ty: Type) -> Result<SimValue, SimError> {
    use SimValue::*;
    Ok(match op {
        Opcode::Ineg => match a {
            I8(x) => I8(x.wrapping_neg()),
            I16(x) => I16(x.wrapping_neg()),
            I32(x) => I32(x.wrapping_neg()),
            I64(x) => I64(x.wrapping_neg()),
            _ => return Err(SimError::TypeMismatch("ineg on float".into())),
        },
        Opcode::Fneg => match a {
            F32(x) => F32(-x),
            F64(x) => F64(-x),
            _ => return Err(SimError::TypeMismatch("fneg on int".into())),
        },
        Opcode::Bnot => match a {
            I8(x) => I8(!x),
            I16(x) => I16(!x),
            I32(x) => I32(!x),
            I64(x) => I64(!x),
            _ => return Err(SimError::TypeMismatch("bnot on float".into())),
        },
        Opcode::Uextend | Opcode::Sextend => {
            // `as_u64` sign-extends narrow types (it routes through `as i64
            // as u64`), so for `uextend` we must explicitly zero-extend
            // from the operand's actual width.
            let raw = match (op, a) {
                (Opcode::Sextend, _) => a.as_i64(),
                (Opcode::Uextend, I8(x)) => x as u8 as i64,
                (Opcode::Uextend, I16(x)) => x as u16 as i64,
                (Opcode::Uextend, I32(x)) => x as u32 as i64,
                (Opcode::Uextend, I64(x)) => x,
                _ => return Err(SimError::TypeMismatch(format!("{op} on float"))),
            };
            match result_ty {
                types::I8 => I8(raw as i8),
                types::I16 => I16(raw as i16),
                types::I32 => I32(raw as i32),
                types::I64 => I64(raw),
                _ => return Err(SimError::TypeMismatch(format!("{op} -> {result_ty}"))),
            }
        }
        Opcode::Ireduce => {
            let raw = a.as_i64();
            match result_ty {
                types::I8 => I8(raw as i8),
                types::I16 => I16(raw as i16),
                types::I32 => I32(raw as i32),
                types::I64 => I64(raw),
                _ => return Err(SimError::TypeMismatch(format!("ireduce -> {result_ty}"))),
            }
        }
        Opcode::Bitcast => {
            let bits = a.as_u64();
            match result_ty {
                types::I8 => I8(bits as i8),
                types::I16 => I16(bits as i16),
                types::I32 => I32(bits as i32),
                types::I64 => I64(bits as i64),
                types::F32 => F32(f32::from_bits(bits as u32)),
                types::F64 => F64(f64::from_bits(bits)),
                _ => return Err(SimError::TypeMismatch(format!("bitcast -> {result_ty}"))),
            }
        }
        _ => return Err(SimError::UnsupportedOpcode(format!("{op}"))),
    })
}

fn icmp(cond: IntCC, a: SimValue, b: SimValue) -> bool {
    let (xs, ys) = (a.as_i64(), b.as_i64());
    let (xu, yu) = (a.as_u64(), b.as_u64());
    match cond {
        IntCC::Equal => xs == ys,
        IntCC::NotEqual => xs != ys,
        IntCC::SignedLessThan => xs < ys,
        IntCC::SignedLessThanOrEqual => xs <= ys,
        IntCC::SignedGreaterThan => xs > ys,
        IntCC::SignedGreaterThanOrEqual => xs >= ys,
        IntCC::UnsignedLessThan => xu < yu,
        IntCC::UnsignedLessThanOrEqual => xu <= yu,
        IntCC::UnsignedGreaterThan => xu > yu,
        IntCC::UnsignedGreaterThanOrEqual => xu >= yu,
    }
}

fn fcmp(cond: FloatCC, a: SimValue, b: SimValue) -> bool {
    let to_f64 = |v: SimValue| match v {
        SimValue::F32(x) => x as f64,
        SimValue::F64(x) => x,
        _ => f64::NAN,
    };
    let (x, y) = (to_f64(a), to_f64(b));
    let ord = !(x.is_nan() || y.is_nan());
    match cond {
        FloatCC::Ordered => ord,
        FloatCC::Unordered => !ord,
        FloatCC::Equal => ord && x == y,
        FloatCC::NotEqual => !ord || x != y,
        FloatCC::OrderedNotEqual => ord && x != y,
        FloatCC::UnorderedOrEqual => !ord || x == y,
        FloatCC::LessThan => ord && x < y,
        FloatCC::LessThanOrEqual => ord && x <= y,
        FloatCC::GreaterThan => ord && x > y,
        FloatCC::GreaterThanOrEqual => ord && x >= y,
        FloatCC::UnorderedOrLessThan => !ord || x < y,
        FloatCC::UnorderedOrLessThanOrEqual => !ord || x <= y,
        FloatCC::UnorderedOrGreaterThan => !ord || x > y,
        FloatCC::UnorderedOrGreaterThanOrEqual => !ord || x >= y,
    }
}

// ---------- memory helpers ----------

fn load_from(mem: &[u8], addr: usize, ty: Type) -> Result<SimValue, SimError> {
    let size = ty.bytes() as usize;
    let end = addr.checked_add(size).ok_or(SimError::OutOfBounds {
        addr: addr as u64,
        size,
        memory_len: mem.len(),
    })?;
    if end > mem.len() {
        return Err(SimError::OutOfBounds {
            addr: addr as u64,
            size,
            memory_len: mem.len(),
        });
    }
    let bytes = &mem[addr..end];
    Ok(match ty {
        types::I8 => SimValue::I8(bytes[0] as i8),
        types::I16 => SimValue::I16(i16::from_le_bytes(bytes.try_into().unwrap())),
        types::I32 => SimValue::I32(i32::from_le_bytes(bytes.try_into().unwrap())),
        types::I64 => SimValue::I64(i64::from_le_bytes(bytes.try_into().unwrap())),
        types::F32 => SimValue::F32(f32::from_le_bytes(bytes.try_into().unwrap())),
        types::F64 => SimValue::F64(f64::from_le_bytes(bytes.try_into().unwrap())),
        _ => return Err(SimError::TypeMismatch(format!("load type {ty}"))),
    })
}

fn store_to(mem: &mut [u8], addr: usize, val: SimValue) -> Result<(), SimError> {
    let size = val.ty().bytes() as usize;
    let end = addr.checked_add(size).ok_or(SimError::OutOfBounds {
        addr: addr as u64,
        size,
        memory_len: mem.len(),
    })?;
    if end > mem.len() {
        return Err(SimError::OutOfBounds {
            addr: addr as u64,
            size,
            memory_len: mem.len(),
        });
    }
    match val {
        SimValue::I8(x) => mem[addr] = x as u8,
        SimValue::I16(x) => mem[addr..end].copy_from_slice(&x.to_le_bytes()),
        SimValue::I32(x) => mem[addr..end].copy_from_slice(&x.to_le_bytes()),
        SimValue::I64(x) => mem[addr..end].copy_from_slice(&x.to_le_bytes()),
        SimValue::F32(x) => mem[addr..end].copy_from_slice(&x.to_le_bytes()),
        SimValue::F64(x) => mem[addr..end].copy_from_slice(&x.to_le_bytes()),
    }
    Ok(())
}

// ---------- formatting helpers ----------

fn format_extname(name: &ExternalName, func: &Function) -> String {
    name.display(Some(&func.params)).to_string()
}

fn format_call_args(args: &[SimValue], memory: &[u8]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let v = args[i];
        // Heuristic: an i64 followed by another i64 in [1..=64] pointing
        // into memory is probably a (ptr, len) fat pointer. Show a preview.
        if let SimValue::I64(p) = v {
            if let Some(SimValue::I64(l)) = args.get(i + 1).copied() {
                if (1..=64).contains(&l) {
                    let lo = p as u64 as usize;
                    let hi = lo.saturating_add(l as usize);
                    if hi <= memory.len() {
                        let bytes = &memory[lo..hi];
                        if let Ok(s) = std::str::from_utf8(bytes) {
                            parts.push(format!("{v} -> {s:?}"));
                            parts.push(format!("{}", args[i + 1]));
                            i += 2;
                            continue;
                        }
                    }
                }
            }
        }
        parts.push(format!("{v}"));
        i += 1;
    }
    parts.join(", ")
}

fn format_call_results(results: &[Value], dfg: &cranelift_codegen::ir::DataFlowGraph) -> String {
    if results.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = results
            .iter()
            .map(|v| format!("{v}:{}", dfg.value_type(*v)))
            .collect();
        format!(" -> {}", parts.join(", "))
    }
}

// ---------- dump ----------

impl SimResult {
    /// Pretty-print the result to stdout. Same content as
    /// [`SimResult::dump_to`].
    pub fn dump(&self) {
        let _ = self.dump_to(&mut io::stdout().lock());
    }

    /// Write a four-section debug dump: optional execution trace, returns,
    /// SSA register table, and a hexdump-style view of memory.
    pub fn dump_to(&self, w: &mut impl Write) -> io::Result<()> {
        if !self.trace.is_empty() {
            writeln!(w, "=== Trace ===")?;
            for line in &self.trace {
                writeln!(w, "{line}")?;
            }
            writeln!(w)?;
        }

        writeln!(w, "=== Returns ===")?;
        if self.returns.is_empty() {
            writeln!(w, "(none)")?;
        } else {
            for (i, v) in self.returns.iter().enumerate() {
                writeln!(w, "ret[{i}] = {v}")?;
            }
        }
        writeln!(w)?;

        writeln!(w, "=== Registers (SSA) ===")?;
        if self.registers.is_empty() {
            writeln!(w, "(none)")?;
        } else {
            for (val, sv) in &self.registers {
                writeln!(w, "{val:<4} = {sv}")?;
            }
        }
        writeln!(w)?;

        let nonzero = self.memory.iter().filter(|b| **b != 0).count();
        writeln!(
            w,
            "=== Memory ({} bytes, {} non-zero) ===",
            self.memory.len(),
            nonzero
        )?;
        write_hexdump(w, &self.memory)?;

        if let Some(err) = &self.error {
            writeln!(w)?;
            writeln!(w, "=== Halted ===")?;
            writeln!(w, "{err}")?;
        }
        Ok(())
    }
}

/// Hexdump in the classic `hexdump -C` shape: address, 16 bytes of hex
/// split 8+8, then an ASCII gutter. Repeated all-zero rows collapse to a
/// `*` line so big buffers stay readable.
fn write_hexdump(w: &mut impl Write, mem: &[u8]) -> io::Result<()> {
    let mut prev_row: Option<[u8; 16]> = None;
    let mut star_active = false;

    for (offset, chunk) in mem.chunks(16).enumerate() {
        let mut row = [0u8; 16];
        row[..chunk.len()].copy_from_slice(chunk);
        let is_full = chunk.len() == 16;
        let last = offset == mem.len().saturating_sub(1) / 16;

        if is_full && !last {
            if let Some(prev) = prev_row {
                if prev == row {
                    if !star_active {
                        writeln!(w, "*")?;
                        star_active = true;
                    }
                    prev_row = Some(row);
                    continue;
                }
            }
        }
        star_active = false;
        write_hexdump_row(w, offset * 16, chunk)?;
        prev_row = Some(row);
    }
    Ok(())
}

fn write_hexdump_row(w: &mut impl Write, addr: usize, bytes: &[u8]) -> io::Result<()> {
    write!(w, "{addr:08x}: ")?;
    for i in 0..16 {
        if i == 8 {
            write!(w, " ")?;
        }
        match bytes.get(i) {
            Some(b) => write!(w, "{b:02x} ")?,
            None => write!(w, "   ")?,
        }
    }
    write!(w, " |")?;
    for b in bytes {
        let c = if (0x20..0x7f).contains(b) {
            *b as char
        } else {
            '.'
        };
        write!(w, "{c}")?;
    }
    for _ in bytes.len()..16 {
        write!(w, " ")?;
    }
    writeln!(w, "|")
}
