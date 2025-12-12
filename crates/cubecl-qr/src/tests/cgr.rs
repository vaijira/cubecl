use std::fmt::Display;

use cubecl_core::{CubeElement, Runtime, prelude::Float};
use cubecl_matmul::{components::MatmulElems, tune_key::MatmulElemType};
use cubecl_std::tensor::TensorHandle;

use crate::tests::test_utils::{assert_equals_approx, tensorhandler_from_data, transpose_matrix};

// Deterministic Pseudocode for A (m x n, m > n)
fn generate_deterministic_sparse_matrix<F: Float>(
    m: usize,
    n: usize,
    bandwidth: usize, // e.g., bandwidth=3 means up to 3 diagonals above the main diagonal are non-zero
) -> Vec<F> {
    // 1. Initialize the matrix A (m x n) to all zeros.
    let mut a_matrix = vec![F::from_int(0); m * n];

    // 2. Set the bandwidth limit for non-zero elements
    // The matrix A[i, j] will be non-zero if j <= i + band
    let band = bandwidth.min(n - 1);

    // 3. Populate the matrix with structured, deterministic values
    for j in 0..n {
        // Start populating from the main diagonal (i = j) and move up/down
        // We iterate through rows 'i' that can contribute to column 'j'.
        // Since m > n, there will be more rows than columns.

        // Rows start at the earliest possible row index (i=j-band) or 0
        let start_i = if j > band { j - band } else { 0 };

        // Rows end at the last row index 'm - 1'
        let end_i = m;

        for i in start_i..end_i {
            // Only populate non-zero elements within the band, i.e.,
            // where the row is not too far below the column index.
            if i <= j + band {
                // Deterministic Value Pattern:
                // Uses the row and column indices to generate a non-zero value.
                // This ensures every test run uses the same matrix.
                let value =
                    F::from_int(i as i64) + F::new(1.0) + F::from_int(j as i64) * F::new(0.1);

                // To introduce a challenge for stability (without randomness):
                // Modulate the value based on the column index to create large differences
                // in magnitude across the matrix.
                if j % 2 == 0 {
                    // Even columns have large values
                    a_matrix[j * n + i] = value * F::new(1e5);
                } else {
                    // Odd columns have small values
                    a_matrix[j * n + 1] = value * F::new(1e-5);
                }
            }
        }
    }

    return a_matrix;
}

pub fn test_qr_cgr<R: Runtime, F: Float + CubeElement + Display>(device: &R::Device, dim: u32) {
    let client = R::client(device);
    let dim_usize = dim as usize;

    let shape = vec![dim as usize, dim as usize];
    let num_elements = shape.iter().product();
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }
    // let data = generate_deterministic_sparse_matrix::<F>(dim_usize, dim_usize, dim_usize - 1);

    let a = tensorhandler_from_data::<R, F>(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    /*let bytes = client.read_one_tensor(a.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("A Output => {output:?}"); */

    let (mut q_t, r) =
        match crate::launch::<R, F>(&crate::Strategy::CommonGivensRotations, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(_) => (
                TensorHandle::empty(&client, shape.clone(), a.dtype),
                TensorHandle::empty(&client, shape.clone(), a.dtype),
            ),
        };

    let bytes = client.read_one_tensor(q_t.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("Q Output => {output:?}");

    let bytes = client.read_one_tensor(r.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("R Output => {output:?}");

    let q = transpose_matrix(&client, &mut q_t);

    /*let bytes = client.read_one_tensor(q_t.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("Q Transposed Output => {output:?}");*/

    let dtypes = MatmulElems {
        lhs_global: MatmulElemType::new(F::as_type_native_unchecked(), false),
        rhs_global: MatmulElemType::new(F::as_type_native_unchecked(), false),
        acc_global: MatmulElemType::new(F::as_type_native_unchecked(), false),
        lhs_stage: MatmulElemType::new(F::as_type_native_unchecked(), false),
        rhs_stage: MatmulElemType::new(F::as_type_native_unchecked(), false),
        acc_stage: MatmulElemType::new(F::as_type_native_unchecked(), false),
        lhs_register: MatmulElemType::new(F::as_type_native_unchecked(), false),
        rhs_register: MatmulElemType::new(F::as_type_native_unchecked(), false),
        acc_register: MatmulElemType::new(F::as_type_native_unchecked(), false),
    };

    let out: TensorHandle<R> = TensorHandle::empty(&client, shape.clone(), a.dtype);
    cubecl_matmul::kernels::naive::launch::<R>(
        &client,
        cubecl_matmul::MatmulInputHandle::Normal(q),
        cubecl_matmul::MatmulInputHandle::Normal(r),
        &out.as_ref(),
        dtypes,
    )
    .unwrap();

    let bytes = client.read_one_tensor(out.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("Result Output => {output:?}");

    if let Err(e) =
        assert_equals_approx::<R, F>(&client, out.handle, &out.shape, &out.strides, &data, 10e-3)
    {
        panic!("{}", e);
    }
}
