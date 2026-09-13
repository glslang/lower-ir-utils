
# Rejected native signatures

These examples fail during type checking on every target. Type aliases do not
bypass the restriction. Use separate scalar parameters or output pointers.

Packed tuple argument (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (i32, i32);
#[lower_ir_utils::jit_export]
fn invalid(_: Unsupported) {}
```

Reordered tuple argument (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (u8, u64, u16);
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::jit_signature!(&*module; fn(Unsupported));
}
```

Large tuple argument (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (i64, i64, i64);
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::define_jit_fn!(
        module, "invalid",
        lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
        fn(Unsupported), |_, _, _| (),
    );
}
```

Mixed-class tuple argument (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (f64, u64);
#[lower_ir_utils::jit_export]
fn invalid(_: Unsupported) {}
```

Nested tuple argument (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = ((i64, i64), i64);
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::jit_signature!(&*module; fn(Unsupported));
}
```

Tuple with a reference (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (u8, &'static u64, u16);
#[lower_ir_utils::jit_export]
fn invalid(_: Unsupported) {}
```

Small tuple return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (i32, i32);
#[lower_ir_utils::jit_export]
fn invalid() -> Unsupported { panic!() }
```

Large tuple return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (i64, f64, i64);
#[lower_ir_utils::jit_export]
fn invalid() -> Unsupported { panic!() }
```

Large JIT return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (i64, f64, i64);
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::define_jit_fn!(
        module, "invalid",
        lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
        fn() -> Unsupported, |_, _, _| (),
    );
}
```

Reordered tuple return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (u8, &'static u64, u16);
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::jit_signature!(&*module; fn() -> Unsupported);
}
```

Full-width tuple return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = (i64, i64);
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::jit_signature!(&*module; fn() -> Unsupported);
}
```

String parameter (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = &'static str;
#[lower_ir_utils::jit_export]
fn invalid(_: Unsupported) {}
```

Shared slice parameter (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = &'static [u8];
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::jit_signature!(&*module; fn(Unsupported));
}
```

Mutable slice parameter (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = &'static mut [u8];
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::define_jit_fn!(
        module, "invalid",
        lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
        fn(Unsupported), |_, _, _| (),
    );
}
```

String return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = &'static str;
#[lower_ir_utils::jit_export]
fn invalid() -> Unsupported { panic!() }
```

Shared slice return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = &'static [u8];
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::jit_signature!(&*module; fn() -> Unsupported);
}
```

Mutable slice return (unsupported `JitParam`):

```compile_fail,E0277
type Unsupported = &'static mut [u8];
fn build(module: &mut lower_ir_utils::__reexport::cranelift_jit::JITModule) {
    let _ = lower_ir_utils::define_jit_fn!(
        module, "invalid",
        lower_ir_utils::__reexport::cranelift_module::Linkage::Export,
        fn() -> Unsupported, |_, _, _| (),
    );
}
```

Custom mappings require an explicit unsafe implementation and an ABI argument:

```compile_fail,E0200
use lower_ir_utils::{JitParam, __reexport::cranelift_codegen::ir::{AbiParam, Type}};
#[repr(transparent)]
struct Word(u64);
impl JitParam for Word {
    fn push_params(out: &mut Vec<AbiParam>, ptr_ty: Type) {
        <u64 as JitParam>::push_params(out, ptr_ty);
    }
}
```
