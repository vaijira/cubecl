use cubecl::{frontend, prelude::*};
use cubecl_linalg::tensor;
use std::marker::PhantomData;

#[cfg(feature = "cuda")]
use half::f16;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;

impl<R: Runtime, E: Float + CubeElement> Benchmark for QRBench<R, E> {
    type Input = (TensorHandle<R, E>, TensorHandle<R, E>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let data = [
            E::from_int(1),
            E::from_int(1),
            E::from_int(2),
            E::from_int(1),
            E::from_int(2),
            E::from_int(1),
            E::from_int(2),
            E::from_int(1),
            E::from_int(1),
        ];
        let handle = client.create(E::as_bytes(&data));
        let r = TensorHandle::<R, E>::new_contiguous(self.shape.to_vec(), handle);

        let q = TensorHandle::<R, E>::empty(&self.client, self.shape.to_vec());
        tensor::identity::launch(&self.client, &q);
        (q, r)
    }

    fn execute(&self, (q, r): Self::Input) {
        // let _ = qr::launch::<R, E>(&qr::Strategy::CommonGivensRotations, &self.client, &q, &r);

        /*let bytes = self.client.read_one(q.handle.binding());
        let output = E::from_bytes(&bytes);
        println!("Final Q Output => {output:?}");

        let bytes = self.client.read_one(r.handle.binding());
        let output = E::from_bytes(&bytes);
        println!("Final R Output => {output:?}");*/
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "qr-{}-[{},{}]-{}",
            R::name(&client),
            self.shape[0],
            self.shape[1],
            E::as_elem_native_unchecked(),
        )
        .to_lowercase()
    }

    fn num_samples(&self) -> usize {
        1
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }
}

#[allow(dead_code)]
struct QRBench<R: Runtime, E> {
    shape: Vec<usize>,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float + CubeElement>(device: R::Device) {
    let client = R::client(&device);

    // for shape in [vec![16, 16], vec![512, 512], vec![4096, 4096]] {
    for shape in [vec![3, 3]] {
        let bench = QRBench::<R, E> {
            shape,
            client: client.clone(),
            device: device.clone(),
            _e: PhantomData,
        };
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::Device));
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, f16>(Default::default());
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, f32>(Default::default());
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default());
    #[cfg(feature = "wgpu-spirv")]
    run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default());
}
