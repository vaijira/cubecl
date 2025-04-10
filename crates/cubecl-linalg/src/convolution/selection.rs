use cubecl_core::{Runtime, client::ComputeClient, prelude::CubePrimitive};

use super::{
    algorithm::{Algorithm, ImplicitCmmaConv},
    base::ConvolutionProblem,
};
use crate::matmul::{
    components::{
        CompleteStageTiling, MatmulPrecision, MatmulSelection, MatmulSize, stage::STAGE_BUFFERING,
        tile::TileMatmulFamily,
    },
    kernels::matmul::find_instruction_shape,
};

pub struct ConvSelection {
    pub matmul: MatmulSelection,
}

pub trait ConvSelector<A: Algorithm> {
    fn select_kernel<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
    ) -> (A::Selection, A::Input);
}

/// Large m stage size for the usual case where `batch_size * out_h * out_w` is significantly larger
/// than `out_channels`
pub struct Large;
/// Balanced stage size for cases where `batch_size * out_h * out_w` is relatively small and `k` or
/// `out_channels` is relatively large
pub struct Balanced;

type Tile<A> = <A as Algorithm>::TileMatmul;

impl ConvSelector<ImplicitCmmaConv> for Large {
    fn select_kernel<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
    ) -> (
        <ImplicitCmmaConv as Algorithm>::Selection,
        <ImplicitCmmaConv as Algorithm>::Input,
    ) {
        let selection = MatmulSelection {
            tile_shape: find_instruction::<R, Tile<ImplicitCmmaConv>, MP>(client, problem),
            tile_count: MatmulSize { m: 8, n: 4, k: 2 },
            plane_dim,
        };
        let config_input = CompleteStageTiling {
            tile_shape: selection.tile_shape,
            tile_count: selection.tile_count,
        };

        let selection = ConvSelection { matmul: selection };

        (selection, (config_input, STAGE_BUFFERING))
    }
}

impl ConvSelector<ImplicitCmmaConv> for Balanced {
    fn select_kernel<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
    ) -> (
        <ImplicitCmmaConv as Algorithm>::Selection,
        <ImplicitCmmaConv as Algorithm>::Input,
    ) {
        let selection = MatmulSelection {
            tile_shape: find_instruction::<R, Tile<ImplicitCmmaConv>, MP>(client, problem),
            tile_count: MatmulSize { m: 4, n: 2, k: 4 },
            plane_dim,
        };
        let config_input = CompleteStageTiling {
            tile_shape: selection.tile_shape,
            tile_count: selection.tile_count,
        };

        let selection = ConvSelection { matmul: selection };

        (selection, (config_input, STAGE_BUFFERING))
    }
}

fn find_instruction<R: Runtime, TMM: TileMatmulFamily, MP: MatmulPrecision>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &ConvolutionProblem,
) -> MatmulSize {
    let (instruction_m, instruction_n, instruction_k) = find_instruction_shape(
        if TMM::requires_tensor_cores() {
            Some((
                client.properties(),
                (
                    MP::ES::as_elem_native_unchecked(),
                    MP::ES::as_elem_native_unchecked(),
                    MP::EA::as_elem_native_unchecked(),
                ),
            ))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    MatmulSize {
        m: instruction_m as u32,
        n: instruction_n as u32,
        k: instruction_k as u32,
    }
}
