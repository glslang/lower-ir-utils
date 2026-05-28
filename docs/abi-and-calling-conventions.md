# ABI & calling conventions

How Rust types cross the `lower-ir-utils` JIT boundary, and why the
Windows-x64 caveats exist. Every claim below is grounded in the source; the
`file:line` citations point at the current implementation.

`lower-ir-utils` is plumbing over Cranelift's JIT, not a compiler or a runtime.
It lowers Rust types into Cranelift `AbiParam`s/`Value`s; it does **not**
allocate, free, or interpret the bytes behind a pointer. Keeping that boundary
in mind explains most of what follows.

---

## 1. Passing pointers vs. `#[repr(C)]`

**You can pass any pointer. A pointer is just a scalar address.**

At the ABI level a pointer contributes exactly one `AbiParam::new(ptr_ty)`
(where `ptr_ty` is the target's pointer width, e.g. `I64`) and lowers to a
single `iconst` of the address. This holds for `*const T`, `*mut T`, `&T`,
`&mut T`, and `usize`/`isize`:

- params: `src/abi.rs:105-129`
- args:   `src/abi.rs:245-286`

There is **no layout requirement on the pointer itself** — an address is an
address. `#[repr(C)]` is irrelevant to *whether* you can hand the JIT a
pointer.

**`#[repr(C)]` only matters for the pointee, and only sometimes.** It becomes
relevant when the **JIT-generated IR itself dereferences the pointer and reads
fields at fixed offsets**. Rust's default `repr(Rust)` layout is unspecified
(field order and padding can change between compiler versions), so if both the
IR and the host must agree on offsets, the pointee needs `#[repr(C)]` (or
`#[repr(transparent)]`, or offsets pinned some other consistent way). If the
pointer is treated opaquely — passed in, handed back, never dereferenced in IR
— repr does not matter at all.

**The real C-ABI tripwire is passing/returning aggregates _by value_, not
pointers.** See §3 and §5.

---

## 2. Why non-`repr(C)` types (e.g. `HashMap`) are passed by pointer

Passing a `HashMap` by pointer is a *consequence* of it not being `#[repr(C)]`,
not a contradiction. It is the standard **opaque-handle** pattern.

Example (`tests/jit_integration.rs:61-65`):

```rust
#[jit_export]
fn lookup(map_ptr: *const HashMap<String, i64>, key: &str) -> i64 {
    let map = unsafe { &*map_ptr };
    *map.get(key).unwrap_or(&-1)
}
```

There are two sides, and only one of them ever touches the map's internals:

1. **The JIT side never looks inside the `HashMap`.** It receives the address
   as one pointer-sized scalar and forwards it straight to `lookup` (call site:
   `tests/jit_integration.rs:84`). To the generated machine code it's an opaque
   integer handle — nothing in the IR dereferences it, so layout is irrelevant.
2. **`lookup` dereferences it — and that's ordinary `rustc`-compiled Rust**
   (`unsafe { &*map_ptr }`, `map.get(...)`). The same compiler laid out the
   `HashMap`, so it knows the real `repr(Rust)` layout. No cross-ABI layout
   agreement is needed.

By-pointer works *precisely because* the only code that interprets the bytes is
the side that knows the layout. Trying to pass the `HashMap` **by value** would
force the JIT IR and `rustc` to agree on every field offset and the total size
— impossible for an unspecified layout. By-value is exactly where `#[repr(C)]`
becomes mandatory; by-pointer sidesteps it.

This is also why the same test passes a `HashMap` (no `repr(C)`) by pointer but
uses `&'static str` (a fat pointer with a known two-register lowering) when it
actually needs the JIT side to carry the data.

---

## 3. Returning strings / owned data

The blocker for returning a `str` is **allocation ownership, not the ABI**.

Mechanically the ABI is fine: `&str` is a fat pointer and lowers to two scalars
(ptr, len), and `JitParam for &str` pushes two `AbiParam`s
(`src/abi.rs:131-136`). On SysV/AAPCS a `{ptr, len}` 16-byte aggregate comes
back in two registers, so the bytes can cross the boundary.

The problem is **where those bytes live**. The crate has no runtime — no
allocator, arena, GC, or interner. A returned `(ptr, len)` is only meaningful
if the pointee outlives the return *and* something eventually frees it.

- **Works without machinery — returning `&'static str`.** If the result is data
  that already exists at build time, you just return the address of static
  bytes: two `iconst`s of a known address (`src/abi.rs:291`). The JIT can even
  select among a fixed set of static strings. No ownership question because
  nothing was allocated.
- **Needs machinery — returning a freshly *computed* string.** The bytes are
  produced at runtime and need storage that outlives the call plus an owner:
  - **Caller-owned out-buffer** (cleanest): caller allocates, passes `*mut u8`
    + capacity, JIT writes and returns the length. Caller owns/frees. No host
    runtime, unambiguous ownership. The crate already lowers `*mut u8` /
    `&mut [u8]`.
  - **Host allocator + free shims**: `#[jit_export]` functions that
    `Box`/`leak`/`from_raw`, with a matching free function the caller must call.
    A hand-rolled ownership protocol across the boundary.
  - **Opaque handle**: return a `*mut String` (the §2 pattern in reverse) and
    provide host `len`/`as_ptr`/`drop` functions; the JIT never touches the
    bytes.

The crate deliberately won't manufacture-and-own a heap string for you — that
would mean shipping a runtime, which is out of scope.

---

## 4. Calling convention

It is **not** something you pick, and it is **not** classic
fastcall/stdcall. It's the **platform's default C calling convention**, derived
automatically from the module's target.

Both signature-building paths route through `Module::make_signature(...)`:

- the `jit_signature!` macro — `src/macros.rs:15`
- the generated `#[jit_export]` `signature()` — `macros/src/lib.rs:251`

`Module::make_signature()` is `Signature::new(isa.default_call_conv())`, so the
call convention comes straight from the target ISA/triple. The doc comment says
it outright (`src/macros.rs:4`): *"the module supplies both the call convention
and the target pointer type."* No `CallConv` is ever set explicitly on the
production path. (The explicit `CallConv::Fast`/`SystemV` uses in the repo are
in the `sim`/proptest harnesses, e.g. `tests/sim.rs:19`, which don't go through
the macros.)

What `default_call_conv()` resolves to per target:

| Target                  | Cranelift `CallConv`                          |
| ----------------------- | --------------------------------------------- |
| x86_64 Linux / macOS    | `SystemV`                                     |
| aarch64 Linux           | `SystemV` (AAPCS lowering)                    |
| aarch64 macOS           | `AppleAarch64`                                |
| **x86_64 Windows**      | **`WindowsFastcall`** (MS x64 ABI)            |
| **aarch64 Windows**     | **`WindowsFastcall`** (AAPCS-style aggregates) |

Notes:

- **Every Windows target resolves to `WindowsFastcall`, regardless of
  architecture.** target-lexicon maps `OperatingSystem::Windows` to
  `WindowsFastcall` before it looks at the arch, so `aarch64-pc-windows-*`
  reports `WindowsFastcall` too — *not* `SystemV`. The arch still decides the
  actual register-level rules though: x86_64 uses the MS x64 aggregate rules
  (§5), while aarch64 follows AAPCS-style aggregate passing. That split is why
  the fat-pointer caveat below is specific to x86_64 Windows even though both
  Windows targets share the `WindowsFastcall` name.
- "fastcall" appears only in the narrow sense that Cranelift *names* the MS x64
  convention `WindowsFastcall`. **stdcall never appears.** These are all 64-bit
  conventions — classic 32-bit `__fastcall`/`__stdcall` are not in play.
- Pinning the signature to the ISA default is what keeps the JIT-generated
  callee ABI-compatible with the `extern "C"` host functions that `#[jit_export]`
  injects: `extern "C"` *is* the platform C ABI.

---

## 5. What "hidden pointer" means on Windows x64

`WindowsFastcall` does put the first four integer/pointer arguments in RCX, RDX,
R8, R9 in order — that ordering is not in dispute. "Hidden pointer" is **not**
about *which* register is used; it's about *what's in it*: the value in the slot
becomes a pointer to memory rather than the data itself. Two distinct flavors,
both of which still obey the RCX/RDX/R8/R9 ordering.

### Large aggregate *arguments* — passed by reference

MS x64 rule: a struct/union of size exactly **1, 2, 4, or 8 bytes** is passed as
if it were an integer of that size (directly in the register). **Any other
size** — including 16 bytes — is passed as a **pointer to a temporary the caller
allocated** (16-byte aligned), and that pointer occupies the normal register
slot.

`&str` is `{ data: *const u8, len: usize }` = 16 bytes, so:

```
f(s: &str)   →   caller stores {ptr,len} in a temp, puts &temp in RCX   // ONE slot, an address
```

But the crate lowers `&str` to **two** pointer-sized params
(`src/abi.rs:131-136`), so the generated callee reads:

```
RCX → "data"     // actually the address of the temp
RDX → "len"      // actually whatever was left in RDX — garbage
```

That mismatch is what the test comment calls out
(`tests/jit_integration.rs:67-68`).

### Large *return values* — hidden first parameter (sret)

This is "hidden pointer" in the strict sense. A return value larger than 8 bytes
(again, not 1/2/4/8) can't come back in RAX, so the ABI inserts an **implicit
first argument**: the caller allocates the result buffer and passes its address
in **RCX**. Every real argument shifts down to RDX, R8, R9, then the stack. The
callee writes through that pointer and returns the same address in RAX.

```
&str g(int x)  →  RCX = &result_buffer (hidden),  RDX = x
```

So RCX is still used — the ordering holds — but it's consumed by the hidden
return-area pointer, not by `x`.

### Why SysV "just works" and Windows x64 doesn't

SysV classifies a 16-byte two-`INTEGER` aggregate into **two** registers — args
in the next two slots, return in RAX:RDX — which matches the crate's two-`ptr_ty`
lowering exactly. AAPCS (aarch64) does the analogous thing, *including on
Windows*: even though `aarch64-pc-windows-*` reports the `WindowsFastcall`
`CallConv` name (§4), its aggregate rules are AAPCS-style, so a 16-byte `&str`
still rides in two registers and the lowering holds. Only `WindowsFastcall`
**on x86_64** diverges, collapsing the aggregate to a single pointer (arg) or an
sret pointer (return).

That is precisely why the fat-pointer (`&str`, `&[T]`, `&mut [T]`) tests are
`#[ignore]`'d on `x86_64-pc-windows-*` and nowhere else — *not* on aarch64
Windows (`tests/jit_integration.rs:67-69`), as documented in the `abi` module
header (`src/abi.rs:13-21`).
