use cubecl_core::{
    CubeElement, Runtime,
    client::ComputeClient,
    ir::StorageType,
    prelude::{Float, TensorHandleRef},
};

use cubecl_std::tensor::{TensorHandle, identity};

use crate::kernels::{QRSetupError, cgr};

type QRTuple<R> = (TensorHandle<R>, TensorHandle<R>);

/// Define the strategy to use when calling for a QR decomposition.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Performs the QR decomposition using Givens rotations.
    /// Better for sparse matrices and less numerically stable than Householder transformations.
    CommonGivensRotations,
}

fn initialize_cgr<R: Runtime>(
    client: &ComputeClient<R>,
    a: &TensorHandleRef<R>,
    dtype: StorageType,
) -> Result<QRTuple<R>, QRSetupError> {
    if a.shape.len() != 2 || a.shape[0] < a.shape[1] {
        return Err(QRSetupError::InvalidShape);
    }
    let q_shape = vec![a.shape[0], a.shape[0]];
    let mut q = TensorHandle::empty(client, q_shape, dtype);
    q.strides = vec![q.shape[0], 1];
    identity::launch(client, &q);

    let a_bytes = client.read_one(a.handle.clone());
    let a_handle = client.create_from_slice(&a_bytes);
    let r = TensorHandle::<R>::new(a_handle, a.shape.to_vec(), a.strides.to_vec(), dtype);

    Ok((q, r))
}

/// It launches a QR decomposition over a m x n matrix a.
///
/// Specify a strategy for the QR decomposition, the client and the matrix a to decompose.
/// In case of success it will return a tuple with the matrix Q transposed and the matrix R in this order.
pub fn launch<R: Runtime, EG: Float + CubeElement>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
) -> Result<QRTuple<R>, QRSetupError> {
    launch_ref::<R, EG>(strategy, client, &a.as_ref(), a.dtype)
}

/// It launches by ref a QR decomposition over a m x n matrix a.
pub fn launch_ref<R: Runtime, EG: Float + CubeElement>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    a: &TensorHandleRef<R>,
    dtype: StorageType,
) -> Result<QRTuple<R>, QRSetupError> {
    let (q_t, r) = match strategy {
        Strategy::CommonGivensRotations => {
            let (q_t, r) = initialize_cgr(client, a, dtype)?;
            cgr::launch_ref::<R, EG>(client, &q_t.as_ref(), &r.as_ref(), dtype);
            (q_t, r)
        }
    };

    Ok((q_t, r))
}
