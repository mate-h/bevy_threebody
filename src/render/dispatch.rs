//! Decides which kernels run this frame and records the compute pass.

use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::RenderContext,
    },
    shader::ShaderCacheError,
};

use super::pipeline::{FractalBindGroups, FractalPipeline, SHADER_ASSET_PATH};
use crate::params::FractalUniform;
use crate::stats::STATS_LEN;

const WORKGROUP: u32 = 8;

/// Tracks which (generation, level) has been initialised on the GPU.
#[derive(Resource, Default)]
pub struct DispatchPlan {
    seen: Option<(u32, u32)>,
    seen_mode: u32,
}

fn pipelines_ready(pipeline: &FractalPipeline, pipeline_cache: &PipelineCache) -> bool {
    for id in pipeline.all() {
        match pipeline_cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Ok(_) => {}
            CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => return false,
            CachedPipelineState::Err(err) => panic!("assets/{SHADER_ASSET_PATH}:\n{err}"),
            _ => return false,
        }
    }
    true
}

pub fn dispatch(
    mut render_context: RenderContext,
    mut plan: ResMut<DispatchPlan>,
    bind_groups: Option<Res<FractalBindGroups>>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FractalPipeline>,
    uniform: Option<Res<FractalUniform>>,
) {
    let (Some(bind_groups), Some(uniform)) = (bind_groups, uniform) else {
        return;
    };
    if !pipelines_ready(&pipeline, &pipeline_cache) {
        return;
    }
    let p = &uniform.0;
    let key = (p.generation, p.level);
    let init = plan.seen != Some(key);
    // A new generation means a new view: warp the old image into it first. A new level of
    // the same generation keeps the coarser image underneath instead.
    // (Not across a mode switch: the old image is from the other mode.)
    let reproject = init
        && plan.seen.is_some_and(|(g, _)| g != p.generation)
        && plan.seen_mode == p.mode;
    plan.seen = Some(key);
    plan.seen_mode = p.mode;

    let get = |id| pipeline_cache.get_compute_pipeline(id).unwrap();
    let sim_groups = p.res.div_ceil(WORKGROUP);
    let tex_groups = p.tex_size.div_ceil(WORKGROUP);
    let lanes = if p.sensitivity != 0 { 2 } else { 1 };

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("threebody"),
            ..default()
        });
    pass.set_bind_group(0, &bind_groups.0[p.front as usize], &[]);

    if reproject {
        pass.set_pipeline(get(pipeline.reproject));
        pass.dispatch_workgroups(tex_groups, tex_groups, 1);
    }
    if init {
        pass.set_pipeline(get(pipeline.init));
        pass.dispatch_workgroups(sim_groups, sim_groups, lanes);
    }
    if p.done == 0 {
        pass.set_pipeline(get(pipeline.advance));
        pass.dispatch_workgroups(sim_groups, sim_groups, lanes);
    }
    if p.mode == 1 {
        pass.set_pipeline(get(pipeline.shade_scatter));
        pass.dispatch_workgroups(p.cells.div_ceil(WORKGROUP), p.cells.div_ceil(WORKGROUP), 1);
        return;
    }
    pass.set_pipeline(get(pipeline.clear_stats));
    pass.dispatch_workgroups((STATS_LEN as u32).div_ceil(64), 1, 1);
    pass.set_pipeline(get(pipeline.shade));
    pass.dispatch_workgroups(sim_groups, sim_groups, 1);
}
