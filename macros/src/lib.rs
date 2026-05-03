//! Proc-macros for [`lower-ir-utils`](../lower_ir_utils/index.html).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, ItemFn, PatType, ReturnType, Type};

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
///     pub fn declare<M: Module>(module: &mut M) -> FuncId;
///     pub fn call<M, A1, A2>(
///         bcx: &mut FunctionBuilder,
///         module: &mut M,
///         id: FuncId,
///         p1: A1, p2: A2,
///     ) -> Value;          // or Inst when R is unit
/// }
/// ```
///
/// Each `A_i: JitArg`, so users can pass either an already-lowered IR `Value`
/// or a Rust constant (`&'static str`, `i64`, `*const T`, ...).
#[proc_macro_attribute]
pub fn jit_export(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemFn);

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

    let return_type: Option<&Type> = match &input.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => match ty.as_ref() {
            // `-> ()` is the same as no return for our purposes.
            Type::Tuple(t) if t.elems.is_empty() => None,
            other => Some(other),
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

    let sig_return_pushes = match return_type {
        Some(rt) => quote! {
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

    let (call_ret_ty, call_ret_expr) = if return_type.is_some() {
        (
            quote! { ::cranelift_codegen::ir::Value },
            quote! { bcx.inst_results(__inst)[0] },
        )
    } else {
        (
            quote! { ::cranelift_codegen::ir::Inst },
            quote! { __inst },
        )
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

            pub fn register(jb: &mut ::cranelift_jit::JITBuilder) {
                jb.symbol(NAME, symbol_addr());
            }

            pub fn signature<M: ::cranelift_module::Module>(
                module: &M,
            ) -> ::cranelift_codegen::ir::Signature {
                let mut sig = module.make_signature();
                let ptr_ty = module.target_config().pointer_type();
                #(#sig_param_pushes)*
                #sig_return_pushes
                sig
            }

            pub fn declare<M: ::cranelift_module::Module>(
                module: &mut M,
            ) -> ::cranelift_module::FuncId {
                let sig = signature(module);
                module
                    .declare_function(NAME, ::cranelift_module::Linkage::Import, &sig)
                    .expect("declare_function failed")
            }

            #[allow(clippy::too_many_arguments)]
            pub fn call<
                M: ::cranelift_module::Module,
                #(#arg_generics: ::lower_ir_utils::JitArg,)*
            >(
                bcx: &mut ::cranelift_frontend::FunctionBuilder<'_>,
                module: &mut M,
                id: ::cranelift_module::FuncId,
                #(#arg_idents: #arg_generics,)*
            ) -> #call_ret_ty {
                use ::cranelift_codegen::ir::InstBuilder as _;
                let ptr_ty = module.target_config().pointer_type();
                let local = module.declare_func_in_func(id, bcx.func);
                let mut args_buf: ::lower_ir_utils::__reexport::smallvec::SmallVec<
                    [::cranelift_codegen::ir::Value; 8]
                > = ::lower_ir_utils::__reexport::smallvec::SmallVec::new();
                #(#arg_lowers)*
                let __inst = bcx.ins().call(local, &args_buf);
                #call_ret_expr
            }
        }
    };

    TokenStream::from(expanded)
}
