use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    CubeElement, Runtime,
};

use crate::tensor::TensorHandle;

use super::{kernels::cgr, kernels::QRLaunchError};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    CommonGivensRotations,
    Simple,
    #[default]
    Auto,
}

// Call with
//    let r =
//      TensorHandle::<R, E>::new(a.shape.to_vec(), a.strides.to_vec(), a.handle.clone());
//  let q =
//      TensorHandle::<R, E>::eye(client, a.shape.to_vec());
pub fn launch<R: Runtime, EG: Float + CubeElement>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    q: &TensorHandle<R, EG>,
    r: &TensorHandle<R, EG>,
) -> Result<(), QRLaunchError> {
    launch_ref::<R, EG>(strategy, client, &q.as_ref(), &r.as_ref())
}

pub fn launch_ref<R: Runtime, EG: Float + CubeElement>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    q: &TensorHandleRef<R>,
    r: &TensorHandleRef<R>,
) -> Result<(), QRLaunchError> {
    match strategy {
        Strategy::CommonGivensRotations => cgr::launch_ref::<R, EG>(client, q, r),
        Strategy::Simple => unimplemented!("missing"),
        Strategy::Auto => unimplemented!("misssing"),
    }
    Ok(())
}
