//! Render-world wiring for the compute pipeline.

mod dispatch;
mod pipeline;

pub use pipeline::OUTPUT_FORMAT;

use bevy::{
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        renderer::RenderGraph,
        storage::ShaderBuffer,
    },
};

use crate::params::FractalUniform;

/// GPU resources owned by the main world (so they can be displayed and read back).
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct FractalTargets {
    /// Ping-pong output textures; `GpuParams::front` is the one being written and shown.
    pub images: [Handle<Image>; 2],
    /// Stats header + 64x64 fate grid, read back every frame.
    pub stats: Handle<ShaderBuffer>,
    /// Per-pixel `info` words of the current level, read back on demand.
    pub fates: Handle<ShaderBuffer>,
}

pub struct FractalComputePlugin;

impl Plugin for FractalComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<FractalTargets>::default(),
            ExtractResourcePlugin::<FractalUniform>::default(),
        ));
        app.sub_app_mut(RenderApp)
            .init_resource::<dispatch::DispatchPlan>()
            .add_systems(RenderStartup, pipeline::init_pipeline)
            .add_systems(
                Render,
                pipeline::prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderGraph, dispatch::dispatch.before(camera_driver));
    }
}
