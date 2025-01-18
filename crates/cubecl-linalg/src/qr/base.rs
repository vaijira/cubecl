use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    CubeElement, Runtime,
};

use crate::tensor::{identity, TensorHandle};

use super::kernels::{cgr, mgs, QRLaunchError};

type QRTuple<R, EG> = (TensorHandle<R, EG>, TensorHandle<R, EG>);

/// Define the strategy to use when making a QR decomposition.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Performs the QR decomposition using the Gram-Schmidt Process.
    /// Numerically unstable, use only for small matrices.
    /// q must contain a copy of the original matrix with dimensions m x n t decompose.
    /// r must contain space for m x n result.
    ModifiedGramSchmidt,
    /// Performs the QR decomposition using Givens rotations.
    /// Good for sparse matrix. Less numerically stable than Householder transformations.
    /// q must contain the identity matrix with dimensions m x m.
    /// r must contain a copy of the original matrix with dimensions m x n to decompose.
    CommonGivensRotations,
}

// Call with
//    let r =
//      TensorHandle::<R, E>::new(a.shape.to_vec(), a.strides.to_vec(), a.handle.clone());
//  let q =
//      TensorHandle::<R, E>::eye(client, a.shape.to_vec());
/// It launches QR decomposition kernel over a m x n matrix.
pub fn launch<R: Runtime, EG: Float + CubeElement>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    a: &TensorHandle<R, EG>,
) -> Result<QRTuple<R, EG>, QRLaunchError> {
    launch_ref::<R, EG>(strategy, client, &a.as_ref())
}

pub fn launch_ref<R: Runtime, EG: Float + CubeElement>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    a: &TensorHandleRef<R>,
) -> Result<QRTuple<R, EG>, QRLaunchError> {
    let (q, r) = match strategy {
        Strategy::CommonGivensRotations => {
            let q = TensorHandle::<R, EG>::empty(client, a.shape.to_vec());
            identity::launch(client, &q);
            let a_bytes = client.read_one(a.handle.clone().binding());
            let a_handle = client.create(&a_bytes);
            let r = TensorHandle::<R, EG>::new(a.shape.to_vec(), a.strides.to_vec(), a_handle);
            cgr::launch_ref::<R, EG>(client, &q.as_ref(), &r.as_ref());
            (q, r)
        }
        Strategy::ModifiedGramSchmidt => {
            let a_bytes = client.read_one(a.handle.clone().binding());
            let a_handle = client.create(&a_bytes);
            let q = TensorHandle::<R, EG>::new_contiguous(a.shape.into(), a_handle);
            let r = TensorHandle::<R, EG>::zeros(client, a.shape.to_vec());
            mgs::launch_ref::<R, EG>(client, &q.as_ref(), &r.as_ref());
            (q, r)
        }
    };

    Ok((q, r))
}
