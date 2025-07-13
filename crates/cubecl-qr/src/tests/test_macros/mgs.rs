#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_qr_mgs {
    () => {
        mod mgs {
            $crate::testgen_qr_mgs!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::tensor::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                cubecl_linalg::qr::tests::mgs::test_mgs::<TestRuntime, FloatT>(&Default::default(), 3);
            }

    };
    ([$($float:ident),*]) => {
        mod mgs {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_qr_mgs!($float);
                })*
            }
        }
    };
}
