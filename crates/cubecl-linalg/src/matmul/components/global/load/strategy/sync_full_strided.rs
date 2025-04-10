use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, Quantization};
use crate::matmul::components::stage::{Stage, StridedTilingLayout};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct SyncFullStridedLoading {}

impl LoadingValidation for SyncFullStridedLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_lines = tiling.total_size() / line_size;
        let total_units = config.num_planes() * config.plane_dim();

        if num_stage_lines % total_units != 0 {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl SyncFullLoadingStrategy for SyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);
        let num_stage_lines = tiling.total_size() / line_size;
        let unit_count = config.num_planes() * config.plane_dim();
        let num_loads_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_base_position = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            let line_read = read_view.load_coalesced_in_stage::<G>(
                unit_position * line_size,
                input_ident,
                config,
            );

            stage.as_slice_mut()[unit_position] = match quantization {
                CubeOption::Some(quantization) => quantization.dequantize(line_read),
                CubeOption::None => Line::cast_from(line_read),
            }
        }
    }
}
