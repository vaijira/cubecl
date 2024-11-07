use std::fmt::Display;

use cubecl_core::{prelude::Float, CubeElement, Runtime};

use crate::matmul::kernels::tiling2d;

use super::test_utils::{assert_equals_approx, MatmulTestCase};

pub fn test_matmul_tiling2d_one_cube<R: Runtime, F: Float + CubeElement + Display>(
    device: &R::Device,
) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_tiling2d::<R, F>(case, device);
}

pub fn test_matmul_tiling2d_several_cubes<R: Runtime, F: Float + CubeElement + Display>(
    device: &R::Device,
) {
    let case = MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
    };

    test_tiling2d::<R, F>(case, device);
}

pub fn test_matmul_tiling2d_with_check_bounds<R: Runtime, F: Float + CubeElement + Display>(
    device: &R::Device,
) {
    let case = MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
    };

    test_tiling2d::<R, F>(case, device);
}

pub fn test_matmul_tiling2d_with_batches<R: Runtime, F: Float + CubeElement + Display>(
    device: &R::Device,
) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
    };

    test_tiling2d::<R, F>(case, device);
}

fn test_tiling2d<R: Runtime, F: Float + CubeElement + Display>(
    case: MatmulTestCase,
    device: &R::Device,
) {
    let client = R::client(device);
    let lhs = case.random_lhs::<R, F>(&client);
    let rhs = case.random_rhs::<R, F>(&client);

    let expected = case.matmul_cpu::<R, F>(&lhs, &rhs, &client);

    let out = tiling2d::launch::<R, F>(
        &client,
        lhs,
        rhs,
        case.empty_out(&client),
        Default::default(),
    );

    if let Err(e) = assert_equals_approx::<R, F>(&client, out.handle, &expected, 0.01) {
        panic!("{}", e);
    }
}