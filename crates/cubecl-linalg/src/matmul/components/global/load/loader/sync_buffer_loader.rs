use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::InputIdent;
use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::single_buffer::BufferReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait SyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        buffer_index: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );
}

#[derive(Clone, CubeType)]
pub struct SyncBufferLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<MP, G, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let stage = Stage::new::<G::SmmConfig>(input_ident.as_ident(), config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncBufferLoader::<MP, G, L> {
            tensor_view,
            stage,
            quantization,
            input_ident,
            _config: PhantomData::<G>,
        }
    }

    pub fn reader(
        this: &Self,
        #[comptime] buffer_id: BufferId,
    ) -> BufferReader<MP::ES, L::TilingLayout> {
        BufferReader::new(this.stage, buffer_id, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] buffer: BufferId, #[comptime] config: G) {
        L::load_buffer::<MP, G>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_index(),
            this.quantization,
            this.input_ident,
            config,
        );
    }
}
