//! Scalar shims must work before, across, and after native register exhaustion.

use cranelift_codegen::ir::InstBuilder;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use lower_ir_utils::{define_jit_fn, jit_export};

macro_rules! boundary_case {
    ($test:ident; $($arg:ident: $ty:ty = $index:literal),+) => {
        mod $test {
            use super::*;

            #[jit_export]
            #[allow(clippy::too_many_arguments)]
            fn host($($arg: $ty,)+ data: *const u8, len: usize, tail: &u64, float: f64) -> u64 {
                // Each lane affects the result; a shift cannot pass unnoticed.
                let bytes = unsafe { std::slice::from_raw_parts(data, len) };
                let mut result = bytes.iter().map(|b| u64::from(*b)).sum::<u64>() + *tail;
                $(result += $arg * ($index + 1);)+
                result + float as u64
            }

            #[test]
            fn round_trip() {
                let mut jb = JITBuilder::new(default_libcall_names()).unwrap();
                host_jit::register(&mut jb);
                let mut module = JITModule::new(jb);
                let ext = host_jit::declare(&mut module);
                let id = define_jit_fn!(
                    &mut module, "boundary", Linkage::Export,
                    fn($($ty,)+ *const u8, usize, &u64, f64) -> u64,
                    |bcx, module, p| {
                        let local = module.declare_func_in_func(ext, bcx.func);
                        let call = bcx.ins().call(local, p);
                        bcx.inst_results(call)[0]
                    },
                ).unwrap();
                module.finalize_definitions().unwrap();
                let f: extern "C" fn($($ty,)+ *const u8, usize, &u64, f64) -> u64 =
                    unsafe { std::mem::transmute(module.get_finalized_function(id)) };
                let bytes = b"scalar ABI";
                let tail = 987_654u64;
                let expected = host($((100 + $index) as $ty,)+ bytes.as_ptr(), bytes.len(), &tail, 42.0);
                assert_eq!(f($((100 + $index) as $ty,)+ bytes.as_ptr(), bytes.len(), &tail, 42.0), expected);
            }
        }
    };
}

// Five preceding words straddled SysV's sixth integer register; seven
// straddled AAPCS64's eighth. Nine also tests both words already on the stack.
boundary_case!(sysv_boundary; a: u64 = 0, b: u64 = 1, c: u64 = 2, d: u64 = 3, e: u64 = 4);
boundary_case!(aapcs_boundary; a: u64 = 0, b: u64 = 1, c: u64 = 2, d: u64 = 3, e: u64 = 4, f: u64 = 5, g: u64 = 6);
boundary_case!(stack_arguments; a: u64 = 0, b: u64 = 1, c: u64 = 2, d: u64 = 3, e: u64 = 4, f: u64 = 5, g: u64 = 6, h: u64 = 7, i: u64 = 8);
