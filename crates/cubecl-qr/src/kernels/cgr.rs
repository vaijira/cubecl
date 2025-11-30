use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::tensor::TensorHandle;

// Followed algorithm described in page 2 and 3 in
// https://thesai.org/Downloads/Volume11No5/Paper_78-Parallel_QR_Factorization_using_Givens_Rotations.pdf

// Fill vector l with the col_index from matrix r.
#[cube(launch, launch_unchecked)]
fn get_column_from_matrix<F: Float>(col_index: u32, r: &Tensor<F>, l: &mut Tensor<F>) {
    if ABSOLUTE_POS < l.len() {
        l[ABSOLUTE_POS] = r[ABSOLUTE_POS + col_index * l.len()]
    }
}

// Execute a Givens rotation of a column.
#[cube(launch, launch_unchecked)]
fn givens_rotation_by_column<F: Float>(
    col_index: u32,
    l: &Tensor<F>,
    q: &mut Tensor<F>,
    r: &mut Tensor<F>,
) {
    if ABSOLUTE_POS < l.len() {
        let col_len = l.len();
        let mut j = col_len - 1;
        let mut mu_prime_i = l[j];
        while j > col_index {
            let a = mu_prime_i;
            mu_prime_i = F::hypot(a, l[j - 1]);
            let c = l[j - 1] / mu_prime_i;
            let s = a / mu_prime_i;

            let j_offset = (col_len * j) + ABSOLUTE_POS;
            let j_1_offset = (col_len * (j - 1)) + ABSOLUTE_POS;

            let mu_i = r[j_1_offset];
            let nu_i = r[j_offset];
            r[j_1_offset] = c * mu_i + s * nu_i;
            r[j_offset] = -s * mu_i + c * nu_i;

            let alpha_i = q[j_1_offset];
            let beta_i = q[j_offset];
            q[j_1_offset] = c * alpha_i + s * beta_i;
            q[j_offset] = -s * alpha_i + c * beta_i;

            j -= 1;
        }
    }
}

/// Launch QR decomposition common Givens rotation kernels by ref
pub fn launch_ref<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) {
    launch::<R, E>(client, q, r, dtype);
}

/// Launch QR decomposition common Givens rotation kernels.
pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) {
    let line_size = 1;

    let cube_dim = CubeDim::default();

    let cube_count = calculate_cube_count_elemwise(r.shape[1] / line_size as usize, cube_dim);

    let l = TensorHandle::<R>::empty(client, [r.shape[1]].to_vec(), dtype);

    for i in 0..r.shape[1] - 1 {
        unsafe {
            let _ = get_column_from_matrix::launch_unchecked::<E, R>(
                client,
                cube_count.clone(),
                cube_dim,
                ScalarArg::new(i as u32),
                r.as_tensor_arg(line_size),
                l.as_ref().as_tensor_arg(line_size),
            );
            let _ = givens_rotation_by_column::launch_unchecked::<E, R>(
                client,
                cube_count.clone(),
                cube_dim,
                ScalarArg::new(i as u32),
                l.as_ref().as_tensor_arg(line_size),
                q.as_tensor_arg(line_size),
                r.as_tensor_arg(line_size),
            );
        }
    }
}
