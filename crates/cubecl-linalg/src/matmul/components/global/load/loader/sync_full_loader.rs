use std::marker::PhantomData;

use crate::matmul::components::global;
use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, single_stage};
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::multi_buffer::FullReader;
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{InputIdent, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait SyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );
}

#[derive(CubeType)]
pub struct SyncFullLoader<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L)>,
}

#[derive(CubeType)]
pub struct SyncFullRhsLoader<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
{
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L)>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(input_ident.as_ident(), config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncFullLoader::<MP, S, L> {
            tensor_view,
            stage,
            quantization,
            input_ident,
            _phantom: PhantomData::<(S, L)>,
        }
    }

    pub fn reader(this: &Self) -> FullReader<MP::ES, L::TilingLayout> {
        FullReader::new(this.stage, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_full::<MP, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage,
            this.quantization,
            this.input_ident,
            config,
        );
    }
}
