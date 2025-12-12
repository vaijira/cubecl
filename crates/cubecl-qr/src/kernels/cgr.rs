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

/*#[cube(launch, launch_unchecked)]
fn givens_rotation_by_column<F: Float>(
    col_index: u32,
    l: &Tensor<F>,
    q: &mut Tensor<F>,
    r: &mut Tensor<F>,
) {
    if ABSOLUTE_POS < r.shape(1) {
        let zero = F::from_int(0);
        let col_len = l.len();
        let mut j = col_len - 1;

        let mut mu_prime_i = f64::cast_from(l[j]);

        while j > col_index {
            let b = f64::cast_from(l[j - 1]);
            let a = mu_prime_i;

            mu_prime_i = f64::hypot(a, b);

            let mut c = f64::from_int(1);
            let mut s = f64::from_int(0);

            if mu_prime_i > f64::EPSILON {
                c = b / mu_prime_i;
                s = a / mu_prime_i;
            }

            // debug_print!("mu_prime_i: %f \n", mu_prime_i);
            let j_offset = (col_len * j) + ABSOLUTE_POS;
            let j_1_offset = (col_len * (j - 1)) + ABSOLUTE_POS;

            if ABSOLUTE_POS == col_index {
                r[j_1_offset] = F::cast_from(mu_prime_i);
                r[j_offset] = zero;
            } else {
                let mu_i = f64::cast_from(r[j_1_offset]);
                let nu_i = f64::cast_from(r[j_offset]);

                r[j_1_offset] = F::cast_from(c * mu_i + s * nu_i);
                r[j_offset] = F::cast_from(-s * mu_i + c * nu_i);
            }

            let alpha_i = f64::cast_from(q[j_1_offset]);
            let beta_i = f64::cast_from(q[j_offset]);

            q[j_1_offset] =  F::cast_from(c * alpha_i + s * beta_i);
            q[j_offset] = F::cast_from(-s * alpha_i + c * beta_i);

            j -= 1;
        }
    }
}
*/

#[cube(launch, launch_unchecked)]
fn zero_lower_triangle<F: Float>(r: &mut Tensor<F>) {
    let num_rows = r.shape(0);
    let num_cols = r.shape(1);
    let row = ABSOLUTE_POS / num_cols;
    let col = ABSOLUTE_POS % num_cols;
    if row > col {
        let offset = num_rows * row + col;
        r[offset] = F::from_int(0);
    }
}

#[cube]
fn givens_rotation<F: Float>(a: F, b: F) -> (F, F) {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let mut c: F = one;
    let mut s: F = zero;
    let abs_a = F::abs(a);
    let abs_b = F::abs(b);

    if abs_a > F::EPSILON {
        if abs_b < F::EPSILON {
            c = zero;
            s = one;
        } else if abs_a > abs_b {
            let r = b / a;
            c = F::rhypot(one, r);
            s = c * r;
        } else {
            let r = a / b;
            s = F::rhypot(one, r);
            c = s * r;
        }
    }
    (c, s)
}

/*
% qrgivens.m
function [Q,R] = qrgivens(A)
  [m,n] = size(A);
  Q = eye(m);
  R = A;

  for j = 1:n
    for i = m:-1:(j+1)
      G = eye(m);
      [c,s] = givensrotation( R(i-1,j),R(i,j) );
      G([i-1, i],[i-1, i]) = [c -s; s c];
      R = G'*R;
      Q = Q*G;
    end
  end

end
*/

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
            //let a = f64::cast_from(l[j]);
            let b = l[j - 1];
            //debug_print!("a: %f, b: %f \n", a, b);
            mu_prime_i = F::hypot(a, b);

            // debug_print!("mu_prime_i: %f \n", mu_prime_i);
            let c = b / mu_prime_i;
            let s = a / mu_prime_i;
            // let (c, s) = givens_rotation::<f64>(a, b);

            //debug_print!("c: %f, s: %f \n", c, s);
            let j_offset = (col_len * j) + ABSOLUTE_POS;
            let j_1_offset = (col_len * (j - 1)) + ABSOLUTE_POS;

            let mu_i = r[j_1_offset];
            let nu_i = r[j_offset];
            r[j_1_offset] = c * mu_i + s * nu_i;
            let r_j_computation = -s * mu_i + c * nu_i;
            if F::abs(r_j_computation) > F::EPSILON {
                r[j_offset] = r_j_computation;
            } else {
                r[j_offset] = F::from_int(0);
            }

            let alpha_i = q[j_1_offset];
            let beta_i = q[j_offset];
            q[j_1_offset] = c * alpha_i + s * beta_i;
            q[j_offset] = -s * alpha_i + c * beta_i;

            j -= 1;
        }
        /*if ABSOLUTE_POS == col_index {
            j = col_len - 1;
            while j > col_index {
                let j_offset = (col_len * j) + ABSOLUTE_POS;
                r[j_offset] = F::from_int(0);
                j -= 1;
            }
        }*/
    }
}

/*
 * https://www.youtube.com/watch?v=B4IHL7j2SRk
 * Steps by processor
 * - - - - - -
 * 4 - - - - -
 * 3 5 - - - -
 * 2 4 6 - - -
 * 1 3 5 7 - -
 * 0 2 4 6 8 -
 */
#[cube(launch, launch_unchecked)]
fn qr_sync_givens_rotation<F: Float>(q: &mut Tensor<F>, r: &mut Tensor<F>) {
    let n_rows = r.shape(0);
    let n_cols = r.shape(1);
    let zero = F::from_int(0);
    let total_steps = (n_cols - 2) * 2;
    let mut current_step = 0;
    let min_step = 2 * ABSOLUTE_POS;
    let max_step = (2 * ABSOLUTE_POS) + (n_cols - ABSOLUTE_POS - 2);
    if ABSOLUTE_POS < n_cols - 1 {
        while current_step <= total_steps {
            if current_step >= min_step && current_step <= max_step {
                let index = n_rows - 1 - (current_step - min_step);
                let j_offset = (index * n_cols) + ABSOLUTE_POS;
                let j_1_offset = ((index - 1) * n_cols) + ABSOLUTE_POS;
                /*debug_print!(
                    "tid: %u current_step: %u min_step: %u max_step %u total steps: %u index: %u j_offset: %u\n",
                    ABSOLUTE_POS,
                    current_step,
                    min_step,
                    max_step,
                    total_steps,
                    index,
                    j_offset
                );*/
                let a = r[j_1_offset];
                let b = r[j_offset];
                let (c, s) = givens_rotation::<F>(a, b);
                r[j_1_offset] = (c * a) + (s * b);
                r[j_offset] = zero; // -s * a + c *b;
                let alpha_i = q[j_1_offset];
                let beta_i = q[j_offset];
                q[j_1_offset] = c * alpha_i + s * beta_i;
                q[j_offset] = -s * alpha_i + c * beta_i;
                let mut col_i = 1;
                while col_i + ABSOLUTE_POS < n_cols {
                    /* debug_print!(
                        "updating neighbour column: %u tid: %u current_step: %u min_step: %u max_step %u total steps: %u index: %u j_offset: %u\n",
                        col_i + ABSOLUTE_POS,
                        ABSOLUTE_POS,
                        current_step,
                        min_step,
                        max_step,
                        total_steps,
                        index,
                        j_offset
                    ); */

                    let mu_i = r[j_1_offset + col_i];
                    let nu_i = r[j_offset + col_i];
                    r[j_1_offset + col_i] = c * mu_i + s * nu_i;
                    r[j_offset + col_i] = -s * mu_i + c * nu_i;
                    let alpha_i = q[j_1_offset + col_i];
                    let beta_i = q[j_offset + col_i];
                    q[j_1_offset + col_i] = c * alpha_i + s * beta_i;
                    q[j_offset + col_i] = -s * alpha_i + c * beta_i;
                    /*let r_cell_1 = r[j_1_offset + col_i];
                    let r_cell = r[j_offset + col_i];
                    let q_cell_1 = q[j_1_offset + col_i];
                    let q_cell = q[j_offset + col_i];

                    debug_print!(
                        "tid: %u col_id %u r[j-1] %f r[j] %f q[j-1] %f q[j] %f\n",
                        ABSOLUTE_POS,
                        col_i,
                        r_cell_1,
                        r_cell,
                        q_cell_1,
                        q_cell,
                    );*/
                    col_i += 1;
                }
                if ABSOLUTE_POS > 0 {
                    // Update pending Q steps
                    let mut q_col_offset = 0;
                    while q_col_offset < ABSOLUTE_POS {
                        let alpha_i = q[j_1_offset - 1 - q_col_offset];
                        let beta_i = q[j_offset - 1 - q_col_offset];
                        q[j_1_offset - 1 - q_col_offset] = c * alpha_i + s * beta_i;
                        q[j_offset - 1 - q_col_offset] = -s * alpha_i + c * beta_i;
                        /* let q_cell_1 = q[j_1_offset - 1 - q_index];
                        let q_cell = q[j_offset - 1 - q_index];

                        debug_print!(
                            "tid: %u q_index %u q[j-1] %f q[j] %f\n",
                            ABSOLUTE_POS,
                            q_index,
                            q_cell_1,
                            q_cell,
                        );*/

                        q_col_offset += 1;
                    }
                }
            }
            current_step += 1;
            sync_cube();
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

    let cube_dim = CubeDim::new_1d(Min::min((r.shape[1] - 1) as u32, 256));

    let cube_count = calculate_cube_count_elemwise((r.shape[1] - 1) / line_size as usize, cube_dim);

    let l = TensorHandle::<R>::empty(client, [r.shape[1]].to_vec(), dtype);
    /*unsafe {
        let _ = qr_sync_givens_rotation::launch_unchecked::<E, R>(
            client,
            cube_count.clone(),
            cube_dim,
            q.as_tensor_arg(line_size),
            r.as_tensor_arg(line_size),
        );
    }*/

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
    unsafe {
        let _ = zero_lower_triangle::launch_unchecked::<E, R>(
            client,
            cube_count.clone(),
            cube_dim,
            r.as_tensor_arg(line_size),
        );
    }
}
