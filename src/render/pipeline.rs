//! Bind group layout, pipelines, buffers and bind groups.

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{storage_buffer_sized, texture_2d, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderDevice, RenderQueue},
        storage::GpuShaderBuffer,
        texture::GpuImage,
    },
};
use std::borrow::Cow;

use super::FractalTargets;
use crate::params::{FractalUniform, GpuParams, TEX_SIZE};

pub const SHADER_ASSET_PATH: &str = "shaders/threebody.wgsl";
pub const OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Bytes per simulated system (`struct Pixel`).
const PIXEL_BYTES: u64 = 48;
/// Two lanes (normal + sensitivity rerun) at the finest level.
pub const STATES_BYTES: u64 = 2 * (TEX_SIZE as u64) * (TEX_SIZE as u64) * PIXEL_BYTES;

#[derive(Resource)]
pub struct FractalPipeline {
    pub layout: BindGroupLayoutDescriptor,
    pub reproject: CachedComputePipelineId,
    pub init: CachedComputePipelineId,
    pub advance: CachedComputePipelineId,
    pub clear_stats: CachedComputePipelineId,
    pub shade: CachedComputePipelineId,
}

impl FractalPipeline {
    pub fn all(&self) -> [CachedComputePipelineId; 5] {
        [self.reproject, self.init, self.advance, self.clear_stats, self.shade]
    }
}

/// Render-world-only storage for every pixel's simulation state.
#[derive(Resource)]
pub struct StatesBuffer(pub Buffer);

/// Indexed by `GpuParams::front`.
#[derive(Resource)]
pub struct FractalBindGroups(pub [BindGroup; 2]);

pub fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "threebody",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<GpuParams>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_2d(OUTPUT_FORMAT, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);
    let queue = |entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from(entry)),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from(entry)),
            ..default()
        })
    };
    let pipeline = FractalPipeline {
        reproject: queue("reproject"),
        init: queue("init"),
        advance: queue("advance"),
        clear_stats: queue("clear_stats"),
        shade: queue("shade"),
        layout,
    };
    commands.insert_resource(pipeline);

    let states = render_device.create_buffer(&BufferDescriptor {
        label: Some("threebody states"),
        size: STATES_BYTES,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    commands.insert_resource(StatesBuffer(states));
}

pub fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<FractalPipeline>,
    pipeline_cache: Res<PipelineCache>,
    targets: Option<Res<FractalTargets>>,
    uniform: Option<Res<FractalUniform>>,
    states: Res<StatesBuffer>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(targets), Some(uniform)) = (targets, uniform) else {
        return;
    };
    let (Some(img0), Some(img1), Some(stats), Some(fates)) = (
        gpu_images.get(&targets.images[0]),
        gpu_images.get(&targets.images[1]),
        gpu_buffers.get(&targets.stats),
        gpu_buffers.get(&targets.fates),
    ) else {
        return;
    };

    let mut uniform_buffer = UniformBuffer::from(uniform.0);
    uniform_buffer.write_buffer(&render_device, &queue);

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let make = |dst: &GpuImage, src: &GpuImage| {
        render_device.create_bind_group(
            "threebody",
            &layout,
            &BindGroupEntries::sequential((
                &uniform_buffer,
                states.0.as_entire_buffer_binding(),
                stats.buffer.as_entire_buffer_binding(),
                fates.buffer.as_entire_buffer_binding(),
                &dst.texture_view,
                &src.texture_view,
            )),
        )
    };
    commands.insert_resource(FractalBindGroups([make(img0, img1), make(img1, img0)]));
}
