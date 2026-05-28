//! Procedural macros for [lower-ir-utils](https://docs.rs/lower-ir-utils).
//!
//! Prefer depending on **`lower-ir-utils`** for the public API (`#[jit_export]` is
//! re-exported there). Match this crate's version to your `lower-ir-utils` dependency.
//!
//! # Crate name
//!
//! The generated helpers reference the parent crate by its canonical name
//! (`::lower_ir_utils::...`). Renaming the dependency in your `Cargo.toml`
//! (e.g. `my-alias = { package = "lower-ir-utils", ... }`) will break the
//! generated code; keep the canonical name `lower-ir-utils`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{FnArg, ItemFn, PatType, ReturnType, Type, parse_macro_input};

/// Annotate a Rust function so it can be called from JIT-compiled Cranelift IR.
///
/// Generates a sibling module `<fn_name>_jit` exposing helpers that hide the
/// boilerplate of (1) registering the symbol with `JITBuilder`, (2) building
/// the Cranelift `Signature`, (3) declaring the import in the module, (4)
/// declaring a local `FuncRef`, and (5) emitting the call.
///
/// If the function has no explicit ABI, `extern "C"` is added automatically.
///
/// # Generated API
///
/// For `fn foo(p1: T1, p2: T2) -> R` the macro emits, alongside the function:
///
/// ```ignore
/// pub mod foo_jit {
///     pub const NAME: &'static str;
///     pub fn symbol_addr() -> *const u8;
///     pub fn register(jb: &mut JITBuilder);
///     pub fn signature<M: Module>(module: &M) -> Signature;
///     pub fn try_declare<M: Module>(module: &mut M) -> ModuleResult<FuncId>;
///     pub fn declare<M: Module>(module: &mut M) -> FuncId;
///     pub fn call<M, A1, A2>(
///         bcx: &mut FunctionBuilder,
///         module: &mut M,
///         id: FuncId,
///         p1: A1, p2: A2,
///     ) -> /* depends on R — see "Return value of `call`" below */;
/// }
/// ```
///
/// Each `A_i: JitArg`, so users can pass either an already-lowered IR `Value`
/// or a Rust constant (`&'static str`, `i64`, `*const T`, ...).
///
/// # Caller obligations (safety)
///
/// `extern "C"` puts the ABI / Rust-validity contract on the **caller** — here,
/// JIT-emitted code that ends up calling this Rust function. The macro only
/// describes the ABI shape; it does not (and cannot) check that the values the
/// JIT site produces actually uphold Rust's validity invariants. For each
/// parameter type the JIT caller must guarantee:
///
/// - `&T` / `&mut T`: aligned, dereferenceable, point to a valid `T` for the
///   call's duration; `&mut T` must not alias any other live access.
/// - `&str`: pointer + length describe UTF-8 bytes that live for the call.
/// - `&[T]` / `&mut [T]`: pointer + length describe a valid slice of `T`s; the
///   `&mut` form must not alias any other live access.
/// - `bool`: the byte passed in must be exactly `0` or `1` (Rust UB otherwise).
/// - Raw pointers (`*const T`, `*mut T`): the pointee must outlive every JIT
///   invocation when the pointer is embedded as an IR immediate; see the
///   `JitArg` impls in `lower_ir_utils::abi`.
///
/// Mismatches surface as Cranelift verifier errors at best and Rust UB at
/// worst. Treat the boundary as you would any other `extern "C"` boundary.
///
/// # Limitations
///
/// **`async fn` is rejected with a compile error.** An `async fn` returns an
/// opaque `impl Future`, not its written output type, so the generated
/// signature — derived from the syntactic return type — would describe the
/// wrong ABI, and JIT machine code has no executor to poll the future anyway.
/// Note that `async extern "C" fn` *does* compile (it only trips
/// `improper_ctypes_definitions`, which this macro silences), so without the
/// explicit rejection the mismatch would surface as UB at run time rather than
/// at compile time. Keep the async work on the host and expose a *synchronous*
/// shim that drives the future to completion (e.g. via
/// `tokio::runtime::Handle::block_on`), then annotate that shim with
/// `#[jit_export]`. See the crate-level "Using in an async runtime (tokio)"
/// docs for the full pattern.
///
/// # Panics
///
/// The generated `declare` helper unwraps `declare_function` with `expect`. It will
/// panic if the symbol is already declared under the same name or if the module rejects
/// the declaration for another reason. Use the generated `try_declare` if you need to
/// surface the error instead.
///
/// # Return value of `call`
///
/// The shape of `call`'s return depends on the annotated function's return type:
///
/// - `-> ()` (or no return): returns `cranelift_codegen::ir::Inst` — the call
///   instruction handle, useful for side-effect-only calls.
/// - Single non-unit return (e.g. `-> i64`, `-> &str`): returns
///   `cranelift_codegen::ir::Value`, the callee's first SSA result.
/// - Tuple return `(T1, ..., TN)`: returns `cranelift_codegen::ir::Inst`. Use
///   `bcx.inst_results(inst)` to get all SSA results in declaration order. The
///   number of results is set by `JitParam::push_params` and may differ from
///   the tuple's element count — fat-pointer types (`&str`, `&[T]`) and
///   nested tuples each push more than one `AbiParam`, so the macro cannot
///   give you a fixed-arity array shape that's correct for every composition.
#[proc_macro_attribute]
pub fn jit_export(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    expand_jit_export(input).into()
}

// Core expansion, split out so it can be unit-tested: it works entirely in
// `proc_macro2` terms (a `proc_macro::TokenStream` can't be constructed outside
// the compiler). Returns either the generated helper module or a
// `compile_error!` invocation.
fn expand_jit_export(mut input: ItemFn) -> TokenStream2 {
    // Reject `async fn`. `async extern "C" fn` actually compiles (it only trips
    // `improper_ctypes_definitions`, which this macro silences), so without an
    // explicit check the helper would generate a signature for the *output*
    // type while the real fn returns an opaque future by value — a silent ABI
    // mismatch / UB. JIT code is synchronous and has no executor to poll a
    // future regardless. Fail loudly and point at the sync-shim workaround.
    if let Some(async_token) = &input.sig.asyncness {
        return syn::Error::new_spanned(
            async_token,
            "#[jit_export] cannot be applied to an `async fn`: JIT call sites are \
             synchronous machine code with no executor, and the generated signature \
             would describe the future's output type rather than the opaque future \
             the function actually returns. Wrap the async work in a synchronous shim \
             (e.g. `tokio::runtime::Handle::block_on`) and annotate that shim instead.",
        )
        .to_compile_error();
    }

    // Auto-inject `extern "C"` if no ABI was specified.
    if input.sig.abi.is_none() {
        input.sig.abi = Some(syn::parse_quote!(extern "C"));
    }

    // Allow idiomatic Rust types like `&str` in the signature without nagging
    // the user about `improper_ctypes_definitions`. This is fine on platforms
    // where the fat-pointer ABI matches separate (ptr, len) args (e.g. SystemV
    // x86_64); users targeting platforms that disagree should use flat params.
    input.attrs.push(syn::parse_quote!(
        #[allow(improper_ctypes_definitions)]
    ));

    let fn_name = input.sig.ident.clone();
    let fn_name_str = fn_name.to_string();
    let helper_mod = format_ident!("{}_jit", fn_name);

    // Collect param types (skip `self` — `extern "C"` fns don't have it but be defensive).
    let param_types: Vec<&Type> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(PatType { ty, .. }) => Some(ty.as_ref()),
            FnArg::Receiver(_) => None,
        })
        .collect();

    // Three return shapes:
    //   - None: unit / no return — `call` yields the `Inst`.
    //   - Single: one non-tuple, non-unit return — `call` yields a single `Value`.
    //   - Multi: a tuple return — `call` yields the `Inst`, because the proc-macro
    //     can only see syntactic arity (tuple-element count) while the actual
    //     ABI-result count is decided by `JitParam` (e.g. `&str`/`&[T]` push two
    //     AbiParams, nested tuples sum their elements). Returning the `Inst`
    //     lets the caller pull the real values via `bcx.inst_results(inst)` —
    //     correct for any composition.
    enum ReturnShape<'a> {
        Single(&'a Type),
        Multi(&'a Type),
    }

    let return_shape: Option<ReturnShape> = match &input.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => match ty.as_ref() {
            Type::Tuple(t) if t.elems.is_empty() => None,
            Type::Tuple(_) => Some(ReturnShape::Multi(ty.as_ref())),
            other => Some(ReturnShape::Single(other)),
        },
    };

    let arg_idents: Vec<_> = (0..param_types.len())
        .map(|i| format_ident!("p{}", i))
        .collect();
    let arg_generics: Vec<_> = (0..param_types.len())
        .map(|i| format_ident!("A{}", i))
        .collect();

    let sig_param_pushes: Vec<TokenStream2> = param_types
        .iter()
        .map(|ty| {
            quote! {
                <#ty as ::lower_ir_utils::JitParam>::push_params(&mut sig.params, ptr_ty);
            }
        })
        .collect();

    let sig_return_pushes = match &return_shape {
        Some(ReturnShape::Single(rt)) | Some(ReturnShape::Multi(rt)) => quote! {
            <#rt as ::lower_ir_utils::JitParam>::push_params(&mut sig.returns, ptr_ty);
        },
        None => quote! {},
    };

    let arg_lowers: Vec<TokenStream2> = arg_idents
        .iter()
        .map(|id| {
            quote! {
                <_ as ::lower_ir_utils::JitArg>::lower(#id, bcx, ptr_ty, &mut args_buf);
            }
        })
        .collect();

    let (call_ret_ty, call_ret_expr) = match &return_shape {
        None | Some(ReturnShape::Multi(_)) => (
            quote! { ::lower_ir_utils::__reexport::cranelift_codegen::ir::Inst },
            quote! { __inst },
        ),
        Some(ReturnShape::Single(_)) => (
            quote! { ::lower_ir_utils::__reexport::cranelift_codegen::ir::Value },
            quote! { bcx.inst_results(__inst)[0] },
        ),
    };

    let expanded = quote! {
        #input

        #[allow(non_snake_case, non_camel_case_types, dead_code)]
        pub mod #helper_mod {
            use super::*;

            pub const NAME: &'static str = #fn_name_str;

            pub fn symbol_addr() -> *const u8 {
                super::#fn_name as *const u8
            }

            pub fn register(jb: &mut ::lower_ir_utils::__reexport::cranelift_jit::JITBuilder) {
                jb.symbol(NAME, symbol_addr());
            }

            pub fn signature<M: ::lower_ir_utils::__reexport::cranelift_module::Module>(
                module: &M,
            ) -> ::lower_ir_utils::__reexport::cranelift_codegen::ir::Signature {
                let mut sig = module.make_signature();
                let ptr_ty = module.target_config().pointer_type();
                #(#sig_param_pushes)*
                #sig_return_pushes
                sig
            }

            pub fn try_declare<M: ::lower_ir_utils::__reexport::cranelift_module::Module>(
                module: &mut M,
            ) -> ::lower_ir_utils::__reexport::cranelift_module::ModuleResult<
                ::lower_ir_utils::__reexport::cranelift_module::FuncId,
            > {
                let sig = signature(module);
                module.declare_function(
                    NAME,
                    ::lower_ir_utils::__reexport::cranelift_module::Linkage::Import,
                    &sig,
                )
            }

            pub fn declare<M: ::lower_ir_utils::__reexport::cranelift_module::Module>(
                module: &mut M,
            ) -> ::lower_ir_utils::__reexport::cranelift_module::FuncId {
                try_declare(module).expect("declare_function failed")
            }

            #[allow(clippy::too_many_arguments)]
            pub fn call<
                M: ::lower_ir_utils::__reexport::cranelift_module::Module,
                #(#arg_generics: ::lower_ir_utils::JitArg,)*
            >(
                bcx: &mut ::lower_ir_utils::__reexport::cranelift_frontend::FunctionBuilder<'_>,
                module: &mut M,
                id: ::lower_ir_utils::__reexport::cranelift_module::FuncId,
                #(#arg_idents: #arg_generics,)*
            ) -> #call_ret_ty {
                use ::lower_ir_utils::__reexport::cranelift_codegen::ir::InstBuilder as _;
                let ptr_ty = module.target_config().pointer_type();
                let local = module.declare_func_in_func(id, bcx.func);
                let mut args_buf: ::lower_ir_utils::__reexport::smallvec::SmallVec<
                    [::lower_ir_utils::__reexport::cranelift_codegen::ir::Value; 8]
                > = ::lower_ir_utils::__reexport::smallvec::SmallVec::new();
                #(#arg_lowers)*
                let __inst = bcx.ins().call(local, &args_buf);
                #call_ret_expr
            }
        }
    };

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_async_fn() {
        let input: ItemFn = syn::parse_quote! {
            async fn fetch(a: i64) -> i64 { a }
        };
        let out = expand_jit_export(input).to_string();
        assert!(
            out.contains("compile_error"),
            "expected a compile_error, got: {out}"
        );
        assert!(
            out.contains("async fn"),
            "error should mention `async fn`: {out}"
        );
    }

    #[test]
    fn expands_plain_fn() {
        let input: ItemFn = syn::parse_quote! {
            fn add(a: i64, b: i64) -> i64 { a + b }
        };
        let out = expand_jit_export(input).to_string();
        assert!(
            !out.contains("compile_error"),
            "should not reject a plain fn: {out}"
        );
        assert!(
            out.contains("mod add_jit"),
            "expected the helper module: {out}"
        );
    }
}
