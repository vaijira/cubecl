use std::fmt::Display;

use crate::matmul::tests::test_utils::assert_equals_approx;
use crate::tensor::{into_contiguous, TensorHandle};
use crate::{qr, tensor};
use cubecl_core::client::ComputeClient;
use cubecl_core::{prelude::Float, CubeElement, Runtime};

fn tensorhandler_from_data<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: Vec<usize>,
    data: &[F],
) -> TensorHandle<R, F> {
    let handle = client.create(F::as_bytes(data));
    TensorHandle::new_contiguous(shape, handle)
}
fn transpose_matrix<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    matrix: &mut TensorHandle<R, F>,
) -> TensorHandle<R, F> {
    matrix.strides.swap(1, 0);
    matrix.shape.swap(1, 0);

    into_contiguous::<R, F>(client, &matrix.as_ref())
}
pub fn test_cgr<R: Runtime, F: Float + CubeElement + Display>(device: &R::Device, dim: u32) {
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
    let mut q = TensorHandle::<R, F>::empty(&client, shape.clone());
    tensor::identity::launch(&client, &q);
    let r = tensorhandler_from_data::<R, F>(&client, shape.clone(), &data);

    let result = qr::launch::<R, F>(&qr::Strategy::CommonGivensRotations, &client, &q, &r);
    assert!(result.is_ok());

    let bytes = client.read_one(q.handle.clone().binding());
    let output = F::from_bytes(&bytes);
    println!("Q Output => {output:?}");

    let bytes = client.read_one(r.handle.clone().binding());
    let output = F::from_bytes(&bytes);
    println!("R Output => {output:?}");

    let q_t = transpose_matrix(&client, &mut q);
    let bytes = client.read_one(q_t.handle.clone().binding());
    let output = F::from_bytes(&bytes);
    println!("Q Transposed Output => {output:?}");

    let out: TensorHandle<R, F> = TensorHandle::empty(&client, shape.clone());
    crate::matmul::kernels::simple::launch::<R, F>(&client, q_t, r, &out.as_ref()).unwrap();

    let bytes = client.read_one(out.handle.clone().binding());
    let output = F::from_bytes(&bytes);
    println!("Result Output => {output:?}");

    if let Err(e) = assert_equals_approx::<R, F>(&client, out.handle, &data, 10e-4) {
        panic!("{}", e);
    }
}
