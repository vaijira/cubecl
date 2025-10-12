use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::PLANE_DIM_APPROX;

use cubecl_std::tensor::TensorHandle;

// Implementation follows GPU_dbl_blocked_houseqr method implementation
// https://github.com/janverschelde/PHCpack/blob/master/src/GPU/Matrices/dbl_baqr_kernels.cu
// https://www.youtube.com/watch?v=x7Lj7VXAW4g
// https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/5/462/files/2016/08/Kerr_Campbell_Richards_QRD_on_GPUs.pdf?bid=462
// Input:  N is number of tiles,
//         n is size of each tile,
//         M is the numbers of rows, M >= Nn,
//         A is an M-by-Nn matrix.
// Output: Q is an orthogonal M-by-M matrix,
//         R is an M-by-Nn matrix, A = QR.
//
// For k from 1 to N do:
//    Compute Householder vectors for one tile, reduce R_k,k.
//    Define Y, compute W and Y * W^t.
//    Add Q * YW^t to update Q.
//    if k < N, add YW^t*C to update R

// Check too https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/SPQR/Source/SuiteSparseQR.cpp
// Algorithm 980: Sparse QR Factorization on the GPU

#[cube(launch_unchecked)]
fn small_house<F: Float>(
    x0: &Array<F>,
    x1: &Array<F>,
    dim: u32,
    dimline_indexlog2: u32,
    v: &mut Tensor<F>,
    beta: &mut Array<F>,
    #[comptime] shared_memory_size: u32,
) {
    let j = UNIT_POS_X;
    let zero_constant = F::from_int(0);
    let one_constant = F::from_int(1);
    let two_constant = F::from_int(2);

    let mut shared_v = SharedMemory::<F>::new(shared_memory_size);
    let mut product = SharedMemory::<F>::new(shared_memory_size);

    let mut stopflag: bool = false;

    shared_v[j] = x1[j]; // reading of vector into shared memory
    product[j] = shared_v[j] * shared_v[j]; // for the 2-norm computation

    v[j + 1] = shared_v[j]; // copies x to v, in case beta is zero
    if j == 0 {
        v[0] = one_constant;
    }

    sync_cube();
    let mut pow_of_two = 1; // sum reduction
    for _ in 0..dimline_indexlog2 {
        if (j % (pow_of_two * 2)) == 0 && (j + pow_of_two < dim) {
            product[j] = product[j] + product[j + pow_of_two];
        }
        pow_of_two = pow_of_two * 2;
        sync_cube();
    }
    // thread 0 computes the sqrt of the inner product, others wait
    if j == 0 {
        if product[0] == zero_constant {
            // product[0] is sigma of house
            beta[0] = F::from_int(0);
            stopflag = true;
        }
    }
    sync_cube();
    if stopflag {
        terminate!(); // case when sigma is zero
    }
    if j == 0 {
        // thread zero sets beta
        let mu = F::sqrt((x0[0]) * (x0[0]) + product[0]);
        let v0 = if x0[0] <= zero_constant {
            x0[0] - mu
        } else {
            -product[0] / (x0[0] + mu)
        };

        let v0_square = v0 * v0;
        beta[0] = two_constant * v0_square / (product[0] + v0_square);
        product[0] = v0; // v0 needed for normalization
    }
    sync_cube();
    if beta[0] != zero_constant {
        shared_v[j] = shared_v[j] / product[0];
    }
    sync_cube();
    v[j + 1] = shared_v[j];
    if j == 0 {
        v[0] = one_constant;
    }
}

fn launch_small_house<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    cols: u32,
    size_tiles: u32,
    num_tiles: u32,
    col_index: u32,
    rows_1: u32,
    k: u32,
    line_index: u32,
    r: &TensorHandleRef<'_, R>,
    v: &TensorHandleRef<'_, R>,
    beta: &TensorHandleRef<'_, R>,
) {
    let rows_log2 = (rows_1 as f32).log2().ceil() as u32;
    let row_index = col_index * (rows + 1); // start of number in A_h
    let v_rows = rows - k * size_tiles; // dimension of V matrix

    println!(
        "rows: {rows} v_rows: {v_rows} cols: {cols} size_tiles: {size_tiles} num_tiles{num_tiles}"
    );
    println!(
        "k: {k} line_index: {line_index} rows_1 {rows_1} col_index {col_index} row_index {row_index}"
    );

    if line_index > 0 {
        let cube_dim = CubeDim::new_1d((PLANE_DIM_APPROX * PLANE_DIM_APPROX) as u32);
        let cube_count =
            calculate_cube_count_elemwise((line_index / line_size as u32) as usize, cube_dim);
        unsafe {
            cubecl_std::tensor::init::zeros_array::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(
                    &v.handle.clone().offset_start((line_index * v_rows) as u64),
                    line_index as usize,
                    line_size,
                ),
            );
        };
    }
    if rows_1 == 0 {
        let cube_dim = CubeDim::new_1d((PLANE_DIM_APPROX * PLANE_DIM_APPROX) as u32);
        let cube_count = calculate_cube_count_elemwise(1 as usize, cube_dim);
        unsafe {
            cubecl_std::tensor::init::zeros_array::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(
                    &beta.handle.clone().offset_start(line_index as u64),
                    1,
                    1,
                ),
            );
        }
        let cube_dim = CubeDim::new_1d((PLANE_DIM_APPROX * PLANE_DIM_APPROX) as u32);
        let cube_count = calculate_cube_count_elemwise(1 as usize, cube_dim);
        unsafe {
            cubecl_std::tensor::init::ones_array::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&v.handle, 1, 1),
            );
        }
    } else {
        let cube_dim = CubeDim::new_1d((PLANE_DIM_APPROX * PLANE_DIM_APPROX) as u32);
        let cube_count = calculate_cube_count_elemwise(rows_1 as usize, cube_dim);
        unsafe {
            small_house::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&r.handle.clone().offset_start(rows as u64), 1, 1),
                ArrayArg::from_raw_parts::<F>(&r.handle.clone().offset_start(rows_1 as u64), 1, 1),
                ScalarArg::new(rows),
                ScalarArg::new(rows_log2),
                TensorArg::from_raw_parts::<F>(
                    &v.handle
                        .clone()
                        .offset_start(((line_index * v_rows) + line_index) as u64),
                    v.strides,
                    v.shape,
                    line_size,
                ),
                ArrayArg::from_raw_parts::<F>(
                    &beta.handle.clone().offset_start(line_index as u64),
                    1,
                    1,
                ),
                cube_dim.num_elems(),
            );
        }
    }
}

#[cube(launch_unchecked)]
fn small_left_r_update<F: Float>(
    rows: u32,
    cols: u32,
    k: u32,
    r: &mut Tensor<F>,
    v: &mut Tensor<F>,
    beta: &mut Array<F>,
    #[comptime] shared_memory_size: u32,
) {
    let tdx = UNIT_POS_X; // index of thread in block
    let r_offset = k * rows + k;
    let mut r_col_index = 0;
    let mut w = F::from_int(0);
    let mut r_tdx = F::from_int(0);

    let mut shared_v = SharedMemory::<F>::new(shared_memory_size);

    shared_v[tdx] = v[tdx];
    sync_cube();

    for i in 0..rows - k {
        // loop through rows of R
        r_tdx = r[r_offset + i + tdx * rows];
        w = w + r_tdx * shared_v[i];
    }
    w = beta[0] * w;
    sync_cube();
    for i in 0..rows - k {
        // update i-th row of R
        r_col_index = r_offset + i + tdx * rows;
        r_tdx = r[r_col_index];
        r_tdx = r_tdx - shared_v[i] * w;
        sync_cube();
        // changed rows-k into cols-k, where cols = endcol
        if tdx < cols - k {
            r[r_col_index] = r_tdx;
        }
    }
}

fn launch_small_left_r_update<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    size_tiles: u32,
    col_index: u32,
    k: u32,
    line_index: u32,
    r: &TensorHandleRef<'_, R>,
    v: &TensorHandleRef<'_, R>,
    beta: &TensorHandleRef<'_, R>,
) {
    let endcol = (k + 1) * size_tiles; // 1 + last column index in tile
    let v_rows = rows - k * size_tiles; // dimension of V matrix

    // changed second argument cols into endcol
    // to avoid updating the next tile
    // must use rows - col_index instead of cols - col_index
    /*dbl_small_leftRupdate<<<1,nrows-colidx>>>
    (rows,endcol,size_tiles,col_index,A_d,&V_d[L*v_rows+line_index],&beta_d[line_index]);*/
    let cube_dim = CubeDim::new_1d((PLANE_DIM_APPROX * PLANE_DIM_APPROX) as u32);
    let cube_count = calculate_cube_count_elemwise((rows - col_index) as usize, cube_dim);
    unsafe {
        small_left_r_update::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rows),
            ScalarArg::new(endcol),
            ScalarArg::new(k),
            r.as_tensor_arg(1),
            TensorArg::from_raw_parts::<F>(
                &v.handle
                    .clone()
                    .offset_start(((line_index * v_rows) + line_index) as u64),
                v.strides,
                v.shape,
                line_size,
            ),
            ArrayArg::from_raw_parts::<F>(
                &beta.handle.clone().offset_start(line_index as u64),
                1,
                1,
            ),
            cube_dim.num_elems(),
        );
    }
}

#[cube(launch_unchecked)]
fn rt_dot_v<F: Float>(
    rows: u32,
    size_tiles: u32,
    col_index: u32,
    r_offset: u32,
    r: &Tensor<F>,
    v: &Tensor<F>,
    rt_dot_v: &mut Tensor<F>,
) {
    let bdx = CUBE_POS_X;
    let tdx = UNIT_POS_X;
    let idx = bdx * size_tiles + tdx; // thread tdx computes RTv[idx]

    let vdx = idx % rows; // index in v is column in R^T
    let row = idx / rows; // R is stored column-by-column

    let rdx = r_offset + idx + (row + 1) * col_index;

    let v_val = v[vdx];
    let r_val = r[rdx];
    let result = r_val * v_val;

    rt_dot_v[idx] = result;
}

#[cube(launch_unchecked)]
fn sum_beta_rt_dot_v<F: Float>(
    rows: u32,
    beta: &Array<F>,
    rt_dot_v: &Tensor<F>,
    w: &mut Tensor<F>,
) {
    let tdx = UNIT_POS_X; // tdx sums elements on row tdx
    let offset = tdx * rows; // number of rows before current row

    let mut result = F::from_int(0);
    let mut r_val = F::from_int(0);

    for i in 0..rows {
        r_val = rt_dot_v[offset + i];
        result = result + r_val;
    }
    r_val = beta[0];
    w[tdx] = r_val * result;
}

#[cube(launch_unchecked)]
fn medium_sub_v_beta_rt_v<F: Float>(
    rows: u32,
    cols: u32,
    size_tiles: u32,
    k: u32,
    r: &mut Tensor<F>,
    v: &Tensor<F>,
    w: &Tensor<F>,
    #[comptime] shared_memory_size: u32,
) {
    let bdx = CUBE_POS_X;
    let tdx = UNIT_POS_X;
    let r_offset = k * rows + k; // start in r
    let w_index = bdx * size_tiles + tdx; // global thread index 

    let col_dim = cols - k; // number of columns in R
    let bound = col_dim * (rows - k); // bound on r_index
    let row_index = w_index / col_dim; // row index
    let col_index = w_index % col_dim; // column index

    let r_index = r_offset + rows * col_index + row_index;

    let mut shared_w = SharedMemory::new(shared_memory_size); // values in beta*R^T*v
    shared_w[tdx] = w[tdx]; // are less in number than size_tiles
    sync_cube();

    let mut r_w_index = r[r_index]; // number that tdx updates
    let v_value = v[row_index]; // value in Householder vector
    let w_value = shared_w[col_index]; // value in beta*R^T*v

    r_w_index = r_w_index - v_value * w_value; // update R[rowidx,colidx]

    if w_index < bound {
        r[r_index] = r_w_index; // if() takes care of padding
    }
}

fn launch_medium_left_r_update<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    size_tiles: u32,
    col_index: u32,
    k: u32,
    line_index: u32,
    r: &TensorHandleRef<'_, R>,
    v: &TensorHandleRef<'_, R>,
    beta: &TensorHandleRef<'_, R>,
    rt_dot_v: &TensorHandleRef<'_, R>,
    w: &TensorHandleRef<'_, R>,
) {
    let endcol = (k + 1) * size_tiles; // 1 + last column index in tile
    let v_rows = rows - k * size_tiles; // dimension of V matrix
    let nhouse = rows - col_index; // length of Householder vector
    // total number of entries in R that will be modified
    let rt_offset = col_index * rows;
    let dim_rt_dot_v = endcol - col_index;
    let sizenum = (rows - col_index) * dim_rt_dot_v;
    let nbrblocks = sizenum.div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    // 2nd argument: cols -> endcol
    // changed second argument cols into endcol
    // to avoid updating the next tile
    // number of threads must be cols - col_index, not endcol - col_index
    unsafe {
        rt_dot_v::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(nhouse),
            ScalarArg::new(size_tiles),
            ScalarArg::new(col_index),
            ScalarArg::new(rt_offset),
            r.as_tensor_arg(line_size),
            TensorArg::from_raw_parts::<F>(
                &v.handle
                    .clone()
                    .offset_start(((line_index * v_rows) + line_index) as u64),
                v.strides,
                v.shape,
                line_size,
            ),
            rt_dot_v.as_tensor_arg(line_size),
        );
    }
    let cube_dim = CubeDim::new_1d(1);
    let cube_count = calculate_cube_count_elemwise(dim_rt_dot_v as usize, cube_dim);
    unsafe {
        sum_beta_rt_dot_v::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(nhouse),
            ArrayArg::from_raw_parts::<F>(
                &beta.handle.clone().offset_start(line_index as u64),
                1,
                1,
            ),
            rt_dot_v.as_tensor_arg(line_size),
            w.as_tensor_arg(line_size),
        );
    }

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        medium_sub_v_beta_rt_v::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rows),
            ScalarArg::new(endcol),
            ScalarArg::new(size_tiles),
            ScalarArg::new(col_index),
            r.as_tensor_arg(line_size),
            TensorArg::from_raw_parts::<F>(
                &v.handle
                    .clone()
                    .offset_start(((line_index * v_rows) + line_index) as u64),
                v.strides,
                v.shape,
                line_size,
            ),
            w.as_tensor_arg(line_size),
            cube_dim.num_elems(),
        );
    }
}

#[cube(launch_unchecked)]
fn beta_times_v<F: Float>(
    rows: u32,
    size_tiles: u32,
    beta: &Array<F>,
    v: &Tensor<F>,
    w: &mut Tensor<F>,
    #[comptime] shared_memory_size: u32,
) {
    let bdx = CUBE_POS_X; // index of block
    let tdx = UNIT_POS_X; // index of thread in block
    let idx = bdx * size_tiles + tdx; // thread tdx computes W[idx]

    let mut shared_v = SharedMemory::new(shared_memory_size); // to store a slice of V

    shared_v[tdx] = v[idx]; // thread tdx loads the data at the global index

    let result = -beta[0] * shared_v[tdx];

    if idx < rows {
        w[idx] = result;
    }
}

#[cube(launch_unchecked)]
fn initialize_wy_t<F: Float>(
    dim: u32,
    size_tiles: u32,
    v: &Tensor<F>,
    w: &Tensor<F>,
    wy_t: &mut Tensor<F>,
) {
    let bdx = CUBE_POS_X; // index of block
    let tdx = UNIT_POS_X; // index of thread in block
    let idx = bdx * size_tiles + tdx; // global index of the thread
    let row = idx / dim; // row index in YWT
    let col = idx % dim; // column index in YWT

    let v_val = v[col];
    let w_val = w[row];
    let result = v_val * w_val;

    if idx < dim * dim {
        wy_t[idx] = result;
    }
}

#[cube(launch_unchecked)]
fn beta_next_w<F: Float>(
    rows: u32,
    size_tiles: u32,
    beta: &Array<F>,
    v: &Tensor<F>,
    w: &mut Tensor<F>,
    wy_t: &Tensor<F>,
    #[comptime] shared_memory_size: u32,
) {
    let bdx = CUBE_POS_X; // index of block
    let tdx = UNIT_POS_X; // index of thread in block
    let idx = bdx * size_tiles + tdx; // global index of the thread
    let wy_t_offset = idx * rows; // start of idx row in YWT
    let mybeta = beta[0];
    let mut vdx = 0;
    let mut wy_t_val = F::from_int(0);
    let mut v_value = F::from_int(0);

    let mut shared_v = SharedMemory::new(shared_memory_size); // to store a slice of V

    shared_v[tdx] = v[idx]; // thread tdx loads the data at the global index

    sync_cube();
    let mut result = shared_v[tdx]; // thread tdx computes the value at index idx

    for i in 0..rows / size_tiles {
        vdx = i * size_tiles + tdx; // index in V and in YWT
        shared_v[tdx] = v[vdx]; // threads load next szt values

        sync_cube();
        for j in 0..size_tiles {
            // multiply szt values with YWT
            wy_t_val = wy_t[wy_t_offset + i * size_tiles + j]; // YWT is stored row by row
            v_value = shared_v[j];
            result = result + wy_t_val * v_value;
        }
        sync_cube();
    }
    let quot = rows / size_tiles;
    let rest = rows - quot * size_tiles; // remainder to compute

    vdx = quot * size_tiles + tdx; // next index to compute
    shared_v[tdx] = v[vdx];

    for j in 0..rest {
        // rest < szt prevents overflow
        sync_cube();
        wy_t_val = wy_t[wy_t_offset + quot * size_tiles + j];
        v_value = shared_v[j];
        result = result + wy_t_val * v_value;
    }
    result = -mybeta * result;

    if idx < rows {
        w[idx] = result;
    }
}

#[cube(launch_unchecked)]
fn update_wy_t<F: Float>(
    dim: u32,
    size_tiles: u32,
    v: &Tensor<F>,
    w: &Tensor<F>,
    wy_t: &mut Tensor<F>,
) {
    let bdx = CUBE_POS_X; // index of block
    let tdx = UNIT_POS_X; // index of thread in block
    let idx = bdx * size_tiles + tdx; // global index of the thread
    let row = idx / dim; // row index in YWT
    let col = idx % dim; // column index in YWT

    let v_val = v[col];
    let w_val = w[row];
    let mut result = wy_t[idx];

    result = result + v_val * w_val;

    if idx < dim * dim {
        wy_t[idx] = result;
    }
}

fn launch_medium_vb_to_w<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    size_tiles: u32,
    index: u32,
    v: &TensorHandleRef<'_, R>,
    w: &TensorHandleRef<'_, R>,
    wy_t: &TensorHandleRef<'_, R>,
    beta: &TensorHandleRef<'_, R>,
) {
    let rowdim = rows - index * size_tiles;
    let nbrblocks1 = rowdim.div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks1);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        beta_times_v::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rowdim),
            ScalarArg::new(size_tiles),
            beta.as_array_arg(line_size),
            v.as_tensor_arg(line_size),
            w.as_tensor_arg(line_size),
            cube_dim.num_elems(),
        );
    }

    let nbrblocks2 = (rowdim * rowdim).div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks2);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        initialize_wy_t::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rowdim),
            ScalarArg::new(size_tiles),
            v.as_tensor_arg(line_size),
            w.as_tensor_arg(line_size),
            wy_t.as_tensor_arg(line_size),
        );
    }

    for j in 1..size_tiles {
        let cube_dim = CubeDim::new_1d(nbrblocks1);
        let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
        unsafe {
            beta_next_w::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ScalarArg::new(rowdim),
                ScalarArg::new(size_tiles),
                ArrayArg::from_raw_parts::<F>(&beta.handle.clone().offset_start(j as u64), 1, 1),
                TensorArg::from_raw_parts::<F>(
                    &v.handle.clone().offset_start((j * rowdim) as u64),
                    v.strides,
                    v.shape,
                    line_size,
                ),
                TensorArg::from_raw_parts::<F>(
                    &w.handle.clone().offset_start((j * rowdim) as u64),
                    w.strides,
                    w.shape,
                    line_size,
                ),
                wy_t.as_tensor_arg(line_size),
                cube_dim.num_elems(),
            );
        }

        let cube_dim = CubeDim::new_1d(nbrblocks2);
        let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
        unsafe {
            update_wy_t::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ScalarArg::new(rowdim),
                ScalarArg::new(size_tiles),
                TensorArg::from_raw_parts::<F>(
                    &v.handle.clone().offset_start((j * rowdim) as u64),
                    v.strides,
                    v.shape,
                    line_size,
                ),
                TensorArg::from_raw_parts::<F>(
                    &w.handle.clone().offset_start((j * rowdim) as u64),
                    w.strides,
                    w.shape,
                    line_size,
                ),
                wy_t.as_tensor_arg(line_size),
            );
        }
    }
}

#[cube(launch_unchecked)]
fn small_qwy_t<F: Float>(
    dim: u32,
    rowdim: u32,
    size_tiles: u32,
    col_offset: u32,
    q: &Tensor<F>,
    wy_t: &Tensor<F>,
    qwy_t: &mut Tensor<F>,
) {
    let bdx = CUBE_POS_X; // index of block
    let tdx = UNIT_POS_X; // index of thread in block
    let offset = bdx * size_tiles + tdx; // offset in result
    let row = offset / rowdim;
    let col = offset % rowdim; // thread 0 computes QWYT[row][col]

    let mut result = F::from_int(0);

    for k in 0..rowdim {
        // run over rowdim, not just szt
        // coloff shifts by col*row elements
        let a = q[row * dim + col_offset + k]; // row = bdx, if dim == szt, coloff == 0
        let b = wy_t[k * rowdim + col]; // if(dim == szt) then col = tdx
        result = result + a * b;
    }
    sync_cube();
    qwy_t[offset] = result; // no column offset in saving QWYT
}

fn launch_small_qwy_t<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    dim: u32,
    size_tiles: u32,
    index: u32,
    q: &TensorHandleRef<'_, R>,
    wy_t: &TensorHandleRef<'_, R>,
    qwy_t: &TensorHandleRef<'_, R>,
) {
    let col_offset = index * size_tiles;
    let rowdim = dim - col_offset;
    let nbrblocks = (dim * rowdim).div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        small_qwy_t::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(dim),
            ScalarArg::new(rowdim),
            ScalarArg::new(size_tiles),
            ScalarArg::new(col_offset),
            q.as_tensor_arg(line_size),
            wy_t.as_tensor_arg(line_size),
            qwy_t.as_tensor_arg(line_size),
        );
    }
}

#[cube(launch_unchecked)]
fn small_q_update<F: Float>(
    dim: u32,
    rowdim: u32,
    size_tiles: u32,
    col_offset: u32,
    q: &mut Tensor<F>,
    qwy_t: &Tensor<F>,
) {
    let bdx = CUBE_POS_X;
    let tdx = UNIT_POS_X;
    let offset = bdx * size_tiles + tdx; // offset in result
    let row = offset / rowdim;
    let col = offset % rowdim;
    let idx1 = row * dim + col_offset + col;

    let mut a = q[idx1]; // row = bdx, if dim == szt, coloff == 0
    let b = qwy_t[offset]; // if(dim == szt) then col = tdx
    a = a + b;

    sync_cube();
    q[idx1] = a;
}

fn launch_small_q_update<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    dim: u32,
    size_tiles: u32,
    index: u32,
    q: &TensorHandleRef<'_, R>,
    qwy_t: &TensorHandleRef<'_, R>,
) {
    let col_offset = index * size_tiles;
    let rowdim = dim - col_offset;
    let nbrblocks = (dim * rowdim).div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        small_q_update::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(dim),
            ScalarArg::new(rowdim),
            ScalarArg::new(size_tiles),
            ScalarArg::new(col_offset),
            q.as_tensor_arg(line_size),
            qwy_t.as_tensor_arg(line_size),
        );
    }
}

#[cube(launch_unchecked)]
fn small_yw_t<F: Float>(
    rows: u32,
    size_tiles: u32,
    w: &Tensor<F>,
    y: &Tensor<F>,
    yw_t: &mut Tensor<F>,
) {
    let bdx = CUBE_POS_X; // index of block
    let tdx = UNIT_POS_X; // index of thread in block
    let offset = bdx * size_tiles + tdx; // offset in result
    let row = offset / rows;
    let col = offset % rows; // thread 0 computes WYT[row][col]

    let mut result = F::from_int(0);

    for k in 0..size_tiles {
        let a = w[k * rows + row]; // if(nrows == szt) then row = bdx
        let b = y[k * rows + col]; // if(nrows == szt) then col = tdx
        result = result + a * b;
    }
    sync_cube();
    yw_t[offset] = result;
}

fn launch_small_yw_t<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    size_tiles: u32,
    index: u32,
    y: &TensorHandleRef<'_, R>,
    w: &TensorHandleRef<'_, R>,
    yw_t: &TensorHandleRef<'_, R>,
) {
    let rowdim = rows - index * size_tiles;
    let nbrblocks = (rowdim * rowdim).div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        small_yw_t::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rowdim),
            ScalarArg::new(size_tiles),
            w.as_tensor_arg(line_size),
            y.as_tensor_arg(line_size),
            yw_t.as_tensor_arg(line_size),
        );
    }
}

#[cube(launch_unchecked)]
fn small_yw_tc<F: Float>(
    rows: u32,
    rowdim: u32,
    coldim: u32,
    size_tiles: u32,
    row_offset: u32,
    col_offset: u32,
    yw_t: &Tensor<F>,
    c: &Tensor<F>,
    yw_tc: &mut Tensor<F>,
) {
    let bdx = CUBE_POS_X; // bdx*size_tiles done by previous blocks
    let tdx = UNIT_POS_X; // index of thread in block
    let offset = bdx * size_tiles + tdx; // offset in result
    let row = offset / coldim; // 1st thread does YWTC[row][col]
    let col = offset % coldim;
    let col_c_offset = (col_offset + col) * rows + row_offset; // 1st element in C

    let mut result = F::from_int(0);

    for k in 0..rowdim {
        // innermost loop runs over rowdim
        let a = yw_t[row * rowdim + k]; // YWT is stored row by row
        let b = c[col_c_offset + k]; // but C is stored column by column
        result = result + a * b;
    }
    sync_cube();
    yw_tc[(col_offset + col) * rows + (row_offset + row)] = result;
}

fn launch_small_yw_tc<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    cols: u32,
    size_tiles: u32,
    index: u32,
    yw_t: &TensorHandleRef<'_, R>,
    yw_tc: &TensorHandleRef<'_, R>,
    c: &TensorHandleRef<'_, R>,
) {
    let row_offset = index * size_tiles;
    let rowdim = rows - row_offset;
    let col_offset = (index + 1) * size_tiles;
    let coldim = cols - col_offset;
    let nbrblocks = (rowdim * coldim).div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        small_yw_tc::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rows),
            ScalarArg::new(rowdim),
            ScalarArg::new(coldim),
            ScalarArg::new(size_tiles),
            ScalarArg::new(row_offset),
            ScalarArg::new(col_offset),
            yw_t.as_tensor_arg(line_size),
            c.as_tensor_arg(line_size),
            yw_tc.as_tensor_arg(line_size),
        );
    }
}

#[cube(launch_unchecked)]
fn small_r_add_yw_tc<F: Float>(
    rows: u32,
    coldim: u32,
    size_tiles: u32,
    row_offset: u32,
    col_offset: u32,
    r: &mut Tensor<F>,
    yw_tc: &Tensor<F>,
) {
    let bdx = CUBE_POS_X;
    let tdx = UNIT_POS_X;
    let offset = bdx * size_tiles + tdx; // offset in result
    let row = offset / coldim; // thread updates R[row][col]
    let col = offset % coldim;
    let idx = (col_offset + col) * rows + (row_offset + row);

    let mut a = r[idx];
    let b = yw_tc[idx];
    a = a + b;

    sync_cube();
    r[idx] = a;
}

fn launch_small_r_add_yw_tc<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    line_size: u8,
    rows: u32,
    cols: u32,
    size_tiles: u32,
    index: u32,
    r: &TensorHandleRef<'_, R>,
    yw_tc: &TensorHandleRef<'_, R>,
) {
    let row_offset = index * size_tiles;
    let rowdim = rows - row_offset;
    let col_offset = (index + 1) * size_tiles;
    let coldim = cols - col_offset;
    let nbrblocks = (rowdim * coldim).div_ceil(size_tiles);

    let cube_dim = CubeDim::new_1d(nbrblocks);
    let cube_count = calculate_cube_count_elemwise(size_tiles as usize, cube_dim);
    unsafe {
        small_r_add_yw_tc::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ScalarArg::new(rows),
            ScalarArg::new(coldim),
            ScalarArg::new(size_tiles),
            ScalarArg::new(row_offset),
            ScalarArg::new(col_offset),
            r.as_tensor_arg(line_size),
            yw_tc.as_tensor_arg(line_size),
        );
    }
}

/// line_indexaunch
pub fn launch_ref<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
) {
    launch::<R, E>(client, q, r);
}

pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
) {
    let rows = q.shape[0];
    let cols = q.shape[1];
    let size_tiles = client.properties().hardware.plane_size_min as usize;
    let num_tiles = cols / size_tiles;
    let size_house = rows;
    let size_pad = size_tiles as usize; // padding for nonsquare tiles
    let size_v_and_w = size_house * size_tiles;

    let vectorization_factor = 1;
    /* tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        r.shape,
        r.strides,
        1,
    );*/

    let cube_dim = CubeDim::default();
    let lines_x = cols as u32 / vectorization_factor as u32;
    let cube_count_x = lines_x.div_ceil(cube_dim.x);
    let cube_count_y = (rows as u32).div_ceil(cube_dim.y);
    let _cube_count = CubeCount::new_2d(cube_count_x, cube_count_y);

    let beta = TensorHandle::<R, E>::zeros(client, [size_tiles as usize].to_vec());
    let v = TensorHandle::<R, E>::zeros(client, [rows].to_vec());
    let w = TensorHandle::<R, E>::zeros(client, [size_v_and_w + size_pad].to_vec());
    let rt_dot_v = TensorHandle::<R, E>::zeros(client, [size_v_and_w + size_pad].to_vec());
    let brtv = TensorHandle::<R, E>::zeros(client, [size_house + size_pad].to_vec());
    let wy_t = TensorHandle::<R, E>::zeros(client, [rows * rows + size_pad].to_vec());
    let qwy_t = TensorHandle::<R, E>::zeros(client, [rows * rows + size_pad].to_vec());
    let yw_t = TensorHandle::<R, E>::zeros(client, [rows * rows + size_pad].to_vec());
    let yw_tc = TensorHandle::<R, E>::zeros(client, [rows * cols + size_pad].to_vec());

    // k runs over the number of blocks
    for k in 0..num_tiles {
        println!("Tile k = {k} out of {num_tiles}");
        // line runs over the columns in one block
        for line_index in 0..size_tiles {
            let col_index = k * size_tiles + line_index; // index of the current column
            let rows_1 = rows - col_index as usize - 1; // #rows in Householder vector - 1
            launch_small_house::<R, E>(
                client,
                vectorization_factor,
                rows as u32,
                cols as u32,
                size_tiles as u32,
                num_tiles as u32,
                col_index as u32,
                rows_1 as u32,
                k as u32,
                line_index as u32,
                r,
                &v.as_ref(),
                &beta.as_ref(),
            );
            let actual = client.read_one(beta.handle.clone());
            let beta_host = E::from_bytes(&actual);
            if beta_host[col_index as usize] == E::from_int(0) {
                println!("Zero beta detected.");
            } else {
                if rows - col_index as usize <= size_tiles as usize {
                    launch_small_left_r_update::<R, E>(
                        client,
                        vectorization_factor,
                        rows as u32,
                        size_tiles as u32,
                        col_index as u32,
                        k as u32,
                        line_index as u32,
                        r,
                        &v.as_ref(),
                        &beta.as_ref(),
                    );
                } else {
                    launch_medium_left_r_update::<R, E>(
                        client,
                        vectorization_factor,
                        rows as u32,
                        size_tiles as u32,
                        col_index as u32,
                        k as u32,
                        line_index as u32,
                        r,
                        &v.as_ref(),
                        &beta.as_ref(),
                        &rt_dot_v.as_ref(),
                        &brtv.as_ref(),
                    );
                }
            }
        }

        launch_medium_vb_to_w::<R, E>(
            client,
            vectorization_factor,
            rows as u32,
            size_tiles as u32,
            k as u32,
            &v.as_ref(),
            &w.as_ref(),
            &wy_t.as_ref(),
            &beta.as_ref(),
        );
        // update Q, WYT matrix has rows - k*size_tiles instead of rows
        launch_small_qwy_t::<R, E>(
            client,
            vectorization_factor,
            rows as u32,
            size_tiles as u32,
            k as u32,
            q,
            &wy_t.as_ref(),
            &qwy_t.as_ref(),
        );
        launch_small_q_update::<R, E>(
            client,
            vectorization_factor,
            rows as u32,
            size_tiles as u32,
            k as u32,
            q,
            &qwy_t.as_ref(),
        );
        if k < num_tiles - 1 {
            // update R
            launch_small_yw_t::<R, E>(
                client,
                vectorization_factor,
                rows as u32,
                size_tiles as u32,
                k as u32,
                &v.as_ref(),
                &w.as_ref(),
                &yw_t.as_ref(),
            );
            launch_small_yw_tc::<R, E>(
                client,
                vectorization_factor,
                rows as u32,
                cols as u32,
                size_tiles as u32,
                k as u32,
                &yw_t.as_ref(),
                r,
                &yw_tc.as_ref(),
            );
            launch_small_r_add_yw_tc::<R, E>(
                client,
                vectorization_factor,
                rows as u32,
                cols as u32,
                size_tiles as u32,
                k as u32,
                r,
                &yw_tc.as_ref(),
            );
        }
    }
}
