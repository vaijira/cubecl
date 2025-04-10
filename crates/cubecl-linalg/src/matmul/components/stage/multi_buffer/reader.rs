use crate::matmul::components::InputIdent;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::tile::Tile;
use crate::matmul::components::tile::TileConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct FullReader<ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    #[cube(comptime)]
    pub input_ident: InputIdent,
}

pub struct FullReaderFamily;

impl ReaderFamily for FullReaderFamily {
    type Reader<ES: Numeric, T: TilingLayout> = FullReader<ES, T>;
}

#[cube]
impl<ES: Numeric, T: TilingLayout> FullReader<ES, T> {
    pub fn new(stage: Stage<ES, T>, #[comptime] input_ident: InputIdent) -> Self {
        FullReader::<ES, T> { stage, input_ident }
    }

    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage.get_tile::<CommonStageConfig<TC>>(
            row,
            col,
            comptime!(this.input_ident.as_ident()),
            config,
        )
    }
}
