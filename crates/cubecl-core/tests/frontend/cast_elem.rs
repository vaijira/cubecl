use cubecl_core as cubecl;
use cubecl_core::{
    cube,
    frontend::{Cast, Numeric},
};

// From float
#[cube]
pub fn float_to_float(x: f32) {
    let y = x + f32::from_int(2);
    let _ = f32::cast_from(y) + f32::from_int(34);
}

#[cube]
pub fn float_to_int(x: f32) {
    let y = x + f32::from_int(2);
    let _ = i32::cast_from(y) + i32::from_int(34);
}

#[cube]
pub fn float_to_u8(x: f32) {
    let y = x + f32::from_int(2);
    let _ = u8::cast_from(y) + u8::from_int(34);
}

#[cube]
pub fn float_to_u16(x: f32) {
    let y = x + f32::from_int(2);
    let _ = u16::cast_from(y) + u16::from_int(34);
}

#[cube]
pub fn float_to_u32(x: f32) {
    let y = x + f32::from_int(2);
    let _ = u32::cast_from(y) + u32::from_int(34);
}

#[cube]
pub fn float_to_u64(x: f32) {
    let y = x + f32::from_int(2);
    let _ = u64::cast_from(y) + u64::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn float_to_bool(x: f32) {
    let y = x + f32::from_int(2);
    let _ = bool::cast_from(y) || true;
}

// From int
#[cube]
pub fn int_to_float(x: i32) {
    let y = x + i32::from_int(2);
    let _ = f32::cast_from(y) + f32::from_int(34);
}

#[cube]
#[allow(clippy::useless_conversion)]
pub fn int_to_int(x: i32) {
    let y = x + i32::from_int(2);
    let _ = i32::cast_from(y) + i32::from_int(34);
}

#[cube]
pub fn int_to_u32(x: i32) {
    let y = x + i32::from_int(2);
    let _ = u32::cast_from(y) + u32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn int_to_bool(x: i32) {
    let y = x + i32::from_int(2);
    let _ = bool::cast_from(y) || true;
}

// From u16
#[cube]
pub fn u16_to_float(x: u16) {
    let y = x + u16::from_int(2);
    let _ = f32::cast_from(y) + f32::from_int(34);
}

#[cube]
pub fn u16_to_int(x: u16) {
    let y = x + u16::from_int(2);
    let _ = i32::cast_from(y) + i32::from_int(34);
}

#[cube]
#[allow(clippy::useless_conversion)]
pub fn u16_to_u32(x: u16) {
    let y = x + u16::from_int(2);
    let _ = u32::cast_from(y) + u32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn u16_to_bool(x: u16) {
    let y = x + u16::from_int(2);
    let _ = bool::cast_from(y) || true;
}

// From u32
#[cube]
pub fn u32_to_float(x: u32) {
    let y = x + u32::from_int(2);
    let _ = f32::cast_from(y) + f32::from_int(34);
}

#[cube]
pub fn u32_to_int(x: u32) {
    let y = x + u32::from_int(2);
    let _ = i32::cast_from(y) + i32::from_int(34);
}

#[cube]
#[allow(clippy::useless_conversion)]
pub fn u32_to_u32(x: u32) {
    let y = x + u32::from_int(2);
    let _ = u32::cast_from(y) + u32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn u32_to_bool(x: u32) {
    let y = x + u32::from_int(2);
    let _ = bool::cast_from(y) || true;
}

// From bool
#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_float(x: bool) {
    let y = x && false;
    let _ = f32::cast_from(y) + f32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_int(x: bool) {
    let y = x && false;
    let _ = i32::cast_from(y) + i32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_u32(x: bool) {
    let y = x && false;
    let _ = u32::cast_from(y) + u32::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
pub fn bool_to_u64(x: bool) {
    let y = x && false;
    let _ = u64::cast_from(y) + u64::from_int(34);
}

#[cube]
#[allow(clippy::overly_complex_bool_expr)]
#[allow(clippy::useless_conversion)]
pub fn bool_to_bool(x: bool) {
    let y = x && false;
    let _ = bool::cast_from(y) || true;
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        frontend::{CubeContext, CubePrimitive},
        ir::{Elem, Item, UIntKind, Variable},
    };

    macro_rules! cast_test {
        ($name:ident, $module:expr, $from:expr, $to:expr) => {
            #[test]
            fn $name() {
                let mut context = CubeContext::default();

                let x = context.create_local_binding($from);

                $module(&mut context, x.into());
                let scope = context.into_scope();

                assert_eq!(
                    format!("{:?}", scope.operations),
                    inline_macro_ref_cast($from, $to)
                );
            }
        };
    }

    cast_test!(
        cube_float_to_int_test,
        float_to_int::expand,
        Item::new(f32::as_elem()),
        Item::new(i32::as_elem())
    );

    cast_test!(
        cube_float_to_u8_test,
        float_to_u8::expand,
        Item::new(f32::as_elem()),
        Item::new(u8::as_elem())
    );

    cast_test!(
        cube_float_to_u16_test,
        float_to_u16::expand,
        Item::new(f32::as_elem()),
        Item::new(u16::as_elem())
    );

    cast_test!(
        cube_float_to_u32_test,
        float_to_u32::expand,
        Item::new(f32::as_elem()),
        Item::new(u32::as_elem())
    );

    cast_test!(
        cube_float_to_u64_test,
        float_to_u64::expand,
        Item::new(f32::as_elem()),
        Item::new(u64::as_elem())
    );

    cast_test!(
        cube_float_to_bool_test,
        float_to_bool::expand,
        Item::new(f32::as_elem()),
        Item::new(Elem::Bool)
    );

    cast_test!(
        cube_int_to_float_test,
        int_to_float::expand,
        Item::new(i32::as_elem()),
        Item::new(f32::as_elem())
    );

    cast_test!(
        cube_int_to_u32_test,
        int_to_u32::expand,
        Item::new(i32::as_elem()),
        Item::new(u32::as_elem())
    );

    cast_test!(
        cube_int_to_bool_test,
        int_to_bool::expand,
        Item::new(i32::as_elem()),
        Item::new(Elem::Bool)
    );

    cast_test!(
        cube_u16_to_float_test,
        u16_to_float::expand,
        Item::new(u16::as_elem()),
        Item::new(f32::as_elem())
    );

    cast_test!(
        cube_u16_to_int_test,
        u16_to_int::expand,
        Item::new(u16::as_elem()),
        Item::new(i32::as_elem())
    );

    cast_test!(
        cube_u16_to_u32_test,
        u16_to_u32::expand,
        Item::new(u16::as_elem()),
        Item::new(u32::as_elem())
    );

    cast_test!(
        cube_u16_to_bool_test,
        u16_to_bool::expand,
        Item::new(u16::as_elem()),
        Item::new(bool::as_elem())
    );

    cast_test!(
        cube_u32_to_float_test,
        u32_to_float::expand,
        Item::new(u32::as_elem()),
        Item::new(f32::as_elem())
    );

    cast_test!(
        cube_u32_to_int_test,
        u32_to_int::expand,
        Item::new(u32::as_elem()),
        Item::new(i32::as_elem())
    );

    cast_test!(
        cube_u32_to_bool_test,
        u32_to_bool::expand,
        Item::new(u32::as_elem()),
        Item::new(Elem::Bool)
    );

    cast_test!(
        cube_bool_to_float_test,
        bool_to_float::expand,
        Item::new(Elem::Bool),
        Item::new(f32::as_elem())
    );

    cast_test!(
        cube_bool_to_int_test,
        bool_to_int::expand,
        Item::new(Elem::Bool),
        Item::new(i32::as_elem())
    );

    cast_test!(
        cube_bool_to_u32_test,
        bool_to_u32::expand,
        Item::new(Elem::Bool),
        Item::new(u32::as_elem())
    );

    fn inline_macro_ref_cast(from_item: Item, to_item: Item) -> String {
        let mut context = CubeContext::default();
        let x = context.create_local_variable(from_item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y = scope.create_local(to_item);

        match from_item.elem() {
            Elem::Float(_) => cpa!(scope, x = x + 2f32),
            Elem::Int(_) => cpa!(scope, x = x + 2i32),
            Elem::AtomicInt(_) => cpa!(scope, x = x + 2i32),
            Elem::UInt(u) => match u {
                UIntKind::U8 => cpa!(scope, x = x + 2u8),
                UIntKind::U16 => cpa!(scope, x = x + 2u16),
                UIntKind::U32 => cpa!(scope, x = x + 2u32),
                UIntKind::U64 => cpa!(scope, x = x + 2u64),
            },
            Elem::AtomicUInt(_) => cpa!(scope, x = x + 2u32),
            Elem::Bool => cpa!(scope, x = x && false),
        }

        cpa!(scope, y = cast(x));

        match to_item.elem() {
            Elem::Float(_) => cpa!(scope, y = y + 34f32),
            Elem::Int(_) => cpa!(scope, y = y + 34i32),
            Elem::AtomicInt(_) => cpa!(scope, y = y + 34i32),
            Elem::UInt(u) => match u {
                UIntKind::U8 => cpa!(scope, y = y + 34u8),
                UIntKind::U16 => cpa!(scope, y = y + 34u16),
                UIntKind::U32 => cpa!(scope, y = y + 34u32),
                UIntKind::U64 => cpa!(scope, y = y + 34u64),
            },
            Elem::AtomicUInt(_) => cpa!(scope, y = y + 34u32),
            Elem::Bool => cpa!(scope, y = y || true),
        }

        format!("{:?}", scope.operations)
    }
}
