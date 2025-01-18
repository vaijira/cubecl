#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_qr_cgr {
    () => {
        mod cgr {
            $crate::testgen_qr_cgr!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::tensor::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                cubecl_linalg::qr::tests::cgr::test_cgr::<TestRuntime, FloatT>(&Default::default(), 3);
            }

    };
    ([$($float:ident),*]) => {
        mod cgr {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_qr_cgr!($float);
                })*
            }
        }
    };
}
