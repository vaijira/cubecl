use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    MatmulPrecision,
    global::{CopyMechanism, GlobalConfig},
};

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
pub trait FullLoader<MP: MatmulPrecision, G: GlobalConfig>:
    CubeType + 'static + Send + Sync
{
    /// The stage reader which matches the input of the underlying stage matmul.
    type StageReader: CubeType;

    /// Returns a reader for the stage at the current k offset
    fn reader(this: &Self) -> Self::StageReader;

    /// Move the k offset by k_offset
    fn advance_view(this: &mut Self, k_offset: u32);
}

#[cube]
pub trait SyncFullLoader<MP: MatmulPrecision, G: GlobalConfig>: FullLoader<MP, G> {
    /// Fills the stage at the current k offset.
    fn fill_stage(this: &mut Self, #[comptime] config: G);
}

#[cube]
pub trait AsyncFullLoader<MP: MatmulPrecision, G: GlobalConfig>: FullLoader<MP, G> {
    /// Fills the stage at the current k offset.
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: G,
    );

    /// Fills the stage with zeros
    fn clear_stage(this: &mut Self, #[comptime] config: G);
}
