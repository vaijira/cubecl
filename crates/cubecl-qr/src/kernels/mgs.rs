use cubecl::prelude::*;
use cubecl_core as cubecl;

// Implementation follows GPU_real_mgs2qr method implementation
// https://github.com/janverschelde/PHCpack/blob/master/src/GPU/MGS2/mgs2_kernels.cu

// Another implementation
// https://thesai.org/Downloads/IJARAI/Volume4No6/Paper_6-Gram_Schmidt_Process_in_Different_Parallel_Platforms.pdf

#[cube(launch)]
fn msg_small_normalize_reduce<F: Float>(
    q: &mut Tensor<F>,
    r: &mut Tensor<F>,
    pivot: u32,
    rows_log2: u32,
) {
    let rows = q.shape(0);
    let b = CUBE_POS_X;
    let j = UNIT_POS_X;
    let block = b + pivot; // column for reduction w.r.t. pivot
    let i = block * rows + j;
    let unit_pivot_index = pivot * rows + j;
    let mut shared_q = SharedMemory::<F>::new(32 * 32 * 2);
    let mut pivots = SharedMemory::<F>::new(32 * 32);
    let mut prd = SharedMemory::<F>::new(32 * 32);
    let r_dim = q.shape(0) * (q.shape(0) + 1) / 2;
    let r_index = (r_dim - 1) - (pivot * (pivot + 1)) / 2 - (b * (b + 1)) / 2 - b * (pivot + 1);
    debug_print!("r_Index: %u\n", r_index);
    let mut pow_of_two = 1; // sum for the norm

    pivots[j] = q[unit_pivot_index];
    prd[j] = pivots[j] * pivots[j];
    sync_cube();

    for _ in 0..rows_log2 {
        if (j % (pow_of_two * 2)) == 0 && j + pow_of_two < rows {
            prd[j] = prd[j] + prd[j + pow_of_two];
        }
        pow_of_two *= 2;
        sync_cube();
    }
    if j == 0 {
        prd[0] = F::sqrt(prd[0]);
    }
    sync_cube();
    pivots[j] = pivots[j] / prd[0];
    if block == pivot {
        q[unit_pivot_index] = pivots[j];
        if j == 0 {
            r[r_index] = prd[0];
        }
    }
    sync_cube();
    if block != pivot {
        shared_q[j] = q[i];
        shared_q[32 * 32 + j] = pivots[j] * shared_q[j];
        sync_cube();
        pow_of_two = 1; // sum for inner product
        for _ in 0..rows_log2 {
            if j % (pow_of_two * 2) == 0 && j + pow_of_two < rows {
                shared_q[32 * 32 + j] = shared_q[32 * 32 + j] + shared_q[32 * 32 + j + pow_of_two];
            }
            pow_of_two *= 2;
            sync_cube();
        }
        shared_q[j] = shared_q[j] - shared_q[32 * 32] * pivots[j];
        sync_cube();
        q[i] = shared_q[j];
        if j == 0 {
            r[r_index] = shared_q[32 * 32];
        }
    }
}

#[cube(launch_unchecked)]
fn mgs_normalize<F: Float>(
    q: &mut Tensor<F>,
    pivot_norm_l2: &mut Array<F>,
    pivot: u32,
    block_size: u32,
) {
    let index = CUBE_POS_X * block_size + UNIT_POS_X;
    let mut pivots = SharedMemory::<F>::new(32 * 32); // contains pivot column

    if index < q.shape(0) {
        pivots[UNIT_POS_X] = q[pivot * q.shape(0) + index];
        pivots[UNIT_POS_X] = pivots[UNIT_POS_X] / pivot_norm_l2[0];
        q[pivot * q.shape(0) + index] = pivots[UNIT_POS_X];
    }
}

#[cube(launch_unchecked)]
fn mgs_normalize_reduce<F: Float>(
    q: &mut Tensor<F>,
    r: &mut Tensor<F>,
    pivot_norm_l2: &mut Array<F>,
    pivot: u32,
    rounds: u32,
    rounds_log2: u32,
    block_size: u32,
    block_size_log2: u32,
) {
    let b = CUBE_POS_X;
    let j = UNIT_POS_X;
    let rows = q.shape(0);
    let cols = q.shape(1);
    let block = b + pivot;
    let mut q_block_index = 0;

    let mut shared_v = SharedMemory::<F>::new(32 * 32);
    let mut pivots = SharedMemory::<F>::new(32 * 32);
    let mut sums = SharedMemory::<F>::new(32);
    let mut new_pivot_norm_l2 = SharedMemory::<F>::new(1);

    for i in 0..rounds {
        if q_block_index + j >= rows {
            // exclude extra threads in last round
            shared_v[j] = F::from_int(0);
        } else {
            pivots[j] = q[pivot * rows + j + q_block_index];
            shared_v[j] = pivots[j] * pivots[j];
        }
        sync_cube();
        let mut pow_of_two = 1; // partial sums for the norm
        for _ in 0..block_size_log2 {
            if j % (pow_of_two * 2) == 0 && j + pow_of_two < block_size {
                shared_v[j] = shared_v[j] + shared_v[j + pow_of_two];
            }
            pow_of_two *= 2;
            sync_cube();
        }
        if j == 0 {
            sums[i] = shared_v[0];
        }
        sync_cube();
        q_block_index += block_size;
    }

    sync_cube();

    let mut pow_of_two = 1; // sum reduction for the norm
    for _ in 0..rounds_log2 {
        if j % (pow_of_two * 2) == 0 && j + pow_of_two < rounds {
            sums[j] = sums[j] + sums[j + pow_of_two];
        }
        pow_of_two *= 2;
        sync_cube();
    }

    if j == 0 {
        new_pivot_norm_l2[0] = F::sqrt(sums[0]);
        let r_index = pivot * (cols + 1);
        /*debug_print!(
            "Updating r_Index %u for j %u with pivot norm %f\n",
            r_index,
            j,
            sum
        );*/
        r[r_index] = new_pivot_norm_l2[0];
    }

    sync_cube();

    q_block_index = 0;
    for i in 0..rounds {
        // normalize and partial sums for inner product
        if q_block_index + j < rows {
            // exclude extra threads in last round
            if (block == pivot) && (pivot > 0) {
                // delayed normalization
                pivots[j] = q[(pivot - 1) * rows + j + q_block_index];
                pivots[j] = pivots[j] / (pivot_norm_l2[0]);
                q[(pivot - 1) * rows + j + q_block_index] = pivots[j];
            } else {
                // every other block applies normalization to pivot column
                pivots[j] = q[pivot * rows + j + q_block_index];
                pivots[j] = pivots[j] / new_pivot_norm_l2[0];
            }
        }
        if block != pivot {
            // nonpivot blocks make inner product
            if q_block_index + j >= rows {
                shared_v[j] = F::from_int(0);
            } else {
                shared_v[j] = q[block * rows + j + q_block_index];
                shared_v[j] = pivots[j] * shared_v[j];
            }
            sync_cube();
            let mut pow_of_two = 1; // partial sums for inner product
            for _ in 0..block_size_log2 {
                if j % (pow_of_two * 2) == 0 && j + pow_of_two < block_size {
                    shared_v[j] = shared_v[j] + shared_v[j + pow_of_two];
                }
                pow_of_two *= 2;
                sync_cube();
            }
            if j == 0 {
                sums[i] = shared_v[0];
            }
            sync_cube();
        }
        sync_cube();
        q_block_index += block_size;
    }
    if block == pivot {
        pivot_norm_l2[0] = new_pivot_norm_l2[0];
    }
    if block != pivot {
        let mut pow_of_two = 1; // sum reduction for inner product
        for _k in 0..rounds_log2 {
            if (j % (pow_of_two * 2)) == 0 && j + pow_of_two < rounds {
                sums[j] = sums[j] + sums[j + pow_of_two];
            }
            pow_of_two *= 2;
            sync_cube();
        }
        q_block_index = 0;
        for _i in 0..rounds {
            // perform reduction
            if q_block_index + j < rows {
                pivots[j] = q[pivot * rows + j + q_block_index];
                shared_v[j] = q[block * rows + j + q_block_index];
                shared_v[j] -= sums[0] * pivots[j] / new_pivot_norm_l2[0];
                q[block * rows + j + q_block_index] = shared_v[j];
            }
            sync_cube();
            q_block_index += block_size;
        }
        if j == 0 {
            // let sum = sums[0];
            let r_index = pivot * cols + block;
            /*debug_print!(
                "Updating r_Index %u for j %u with norm %f\n",
                r_index,
                j,
                sum
            );*/
            r[r_index] = sums[0];
        }
    }
}

pub fn launch_ref<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
) {
    launch::<R, E>(client, q, r);
}

pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
) {
    let vectorization_factor = 1;
    let block_size = 32;
    let cube_dim = CubeDim::new_1d(block_size);
    let num_elems_per_cube = cube_dim.num_elems();
    let rounds = q.shape[0].div_ceil(num_elems_per_cube as usize) as u32;
    let pivot_norm_l2 = E::from_int(0);
    let handle_pivot_norm_l2 = client.create(E::as_bytes(&[pivot_norm_l2]));

    /*for pivot in 0..q.shape[1] {
        let cube_count = CubeCount::new_1d(3u32); //q.shape[1] as u32 -pivot as u32);
        println!(
             "loop with pivot {} cube_count {}",
             pivot,
             q.shape[1] - pivot,
         );

        msg_small_normalize_reduce::launch::<E, R>(
            client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(vectorization_factor),
            r.as_tensor_arg(vectorization_factor),
            ScalarArg::new(pivot as u32),
            ScalarArg::new((q.shape[0] as f32).log2().ceil() as u32),
        );
    }*/

    for pivot in 0..q.shape[1] {
        let pivot_norm_l2 = unsafe {
            ArrayArg::<R>::from_raw_parts::<E>(&handle_pivot_norm_l2, 1, vectorization_factor)
        };
        let block_size_log2 = block_size - block_size.leading_zeros(); // ceil of log2
        let cube_count = CubeCount::new_1d((q.shape[1] - pivot) as u32);
        /* println!(
            "loop with pivot {} block_size_log2 {} cube_count {} and rounds {}",
            pivot,
            block_size_log2,
            q.shape[1] - pivot,
            rounds
        ); */

        unsafe {
            mgs_normalize_reduce::launch_unchecked::<E, R>(
                client,
                cube_count.clone(),
                cube_dim,
                q.as_tensor_arg(vectorization_factor),
                r.as_tensor_arg(vectorization_factor),
                pivot_norm_l2,
                ScalarArg::new(pivot as u32),
                ScalarArg::new(rounds),
                ScalarArg::new((rounds as f32 + 1.).log2().ceil() as u32),
                ScalarArg::new(block_size),
                ScalarArg::new(block_size_log2),
            );
        }
    }

    let cube_count = CubeCount::new_1d(rounds);
    let pivot_norm_l2 = unsafe {
        ArrayArg::<R>::from_raw_parts::<E>(&handle_pivot_norm_l2, 1, vectorization_factor)
    };
    unsafe {
        mgs_normalize::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(vectorization_factor),
            pivot_norm_l2,
            ScalarArg::new((q.shape[1] - 1) as u32),
            ScalarArg::new(block_size),
        );
    }
    /* let bytes = client.read_one(q.handle.clone().binding());
    let output = E::from_bytes(&bytes);
    println!("Q Output => {output:?}");

    let bytes = client.read_one(r.handle.clone().binding());
    let output = E::from_bytes(&bytes);
    println!("R Output => {output:?}"); */
}
