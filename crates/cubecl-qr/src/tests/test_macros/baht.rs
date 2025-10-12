#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_qr_baht {
    () => {
        mod baht {
            $crate::testgen_qr_baht!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_std::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                cubecl_qr::tests::baht::test_qr_baht::<TestRuntime, FloatT>(&Default::default(), 3);
            }

    };
    ([$($float:ident),*]) => {
        mod baht {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_qr_baht!($float);
                })*
            }
        }
    };
}
