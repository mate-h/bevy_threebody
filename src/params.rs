//! Main-world simulation settings and the uniform sent to the compute shader.

use bevy::{
    math::DVec2,
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource, render_resource::ShaderType},
};
use bevy_threebody::physics_cpu::Config;

/// Side length of the output texture; the finest refinement level simulates one system
/// per texel.
pub const TEX_SIZE: u32 = 1024;
/// Simulation resolution per refinement level, coarse to fine.
pub const LEVELS: [u32; 3] = [256, 512, 1024];
/// Half-width of the default view in world units.
pub const DEFAULT_HALF_WIDTH: f64 = 3.0;

/// Scattering table: cells per side, encounters running per cell (cells^2 * samples must
/// equal TEX_SIZE^2, one system per finest-level pixel), and axis ranges.
pub const SCATTER_CELLS: u32 = 128;
pub const SCATTER_SAMPLES: u32 = 64;
pub const SCATTER_V_MAX: f64 = 3.0;
pub const SCATTER_RP_MAX: f64 = 4.0;
/// Binary eccentricity presets; negative = thermal distribution.
pub const ECC_PRESETS: [f64; 4] = [0.0, 0.5, 0.9, -1.0];

/// Mirrors `struct Params` in `assets/shaders/threebody.wgsl` (same field order).
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct GpuParams {
    pub masses: Vec4,
    pub center: Vec2,
    pub prev_center: Vec2,
    pub pos_a: Vec2,
    pub pos_b: Vec2,
    pub scale: f32,
    pub prev_scale: f32,
    pub eta: f32,
    pub eta_b: f32,
    pub t_max_cr: f32,
    pub r_far: f32,
    pub res: u32,
    pub tex_size: u32,
    pub generation: u32,
    pub level: u32,
    pub steps_per_frame: u32,
    pub max_steps: u32,
    pub color_mode: u32,
    pub sensitivity: u32,
    pub front: u32,
    pub done: u32,
    pub mode: u32,
    pub scatter_ecc: f32,
    pub cells: u32,
    pub samples: u32,
    pub v_max: f32,
    pub rp_max: f32,
    pub seed_base: u32,
}

/// What the render world sees each frame.
#[derive(Resource, Clone, Debug, ExtractResource)]
#[extract_app(RenderApp)]
pub struct FractalUniform(pub GpuParams);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Fate,
    EjectionTime,
    EnergyError,
    Steps,
    Sensitivity,
}

impl ColorMode {
    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Fate => "fate",
            ColorMode::EjectionTime => "ejection time",
            ColorMode::EnergyError => "energy error",
            ColorMode::Steps => "step count",
            ColorMode::Sensitivity => "sensitivity",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimMode {
    /// Every pixel: three stars from rest, coloured by which is ejected.
    FateMap,
    /// Every cell: random binary-single encounters, coloured by outcome probabilities.
    Scattering,
}

/// Mass presets from the video: equal, 3-4-5 and 1-2-3.
pub const MASS_PRESETS: [[f64; 3]; 3] = [[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]];

/// The view and physics settings the user controls.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct Settings {
    pub center: DVec2,
    pub half_width: f64,
    pub physics: Config,
    pub mass_preset: usize,
    /// eta of the second (sensitivity) lane, as a fraction of `physics.eta`.
    pub eta_b_ratio: f64,
    pub sensitivity: bool,
    pub color_mode: ColorMode,
    pub mode: SimMode,
    pub ecc_preset: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            center: DVec2::ZERO,
            half_width: DEFAULT_HALF_WIDTH,
            physics: Config::default(),
            mass_preset: 0,
            eta_b_ratio: 0.5,
            sensitivity: false,
            color_mode: ColorMode::Fate,
            mode: SimMode::FateMap,
            ecc_preset: 0,
        }
    }
}

impl Settings {
    /// World units per output texel.
    pub fn scale(&self) -> f64 {
        2.0 * self.half_width / TEX_SIZE as f64
    }

    /// Smallest half-width at which neighbouring pixels still get distinct f32 starting
    /// positions (with a few ulps of margin). Beyond this an f32 simulation can't tell
    /// pixels apart.
    pub fn min_half_width(&self) -> f64 {
        let mag = (self.center.abs().max_element() + self.half_width).max(1.0);
        let ulp = mag * f32::EPSILON as f64;
        8.0 * ulp * TEX_SIZE as f64 / 2.0
    }

    pub fn zoom_factor(&self) -> f64 {
        DEFAULT_HALF_WIDTH / self.half_width
    }

    /// Settings whose change invalidates every pixel.
    fn sim_key(&self) -> (DVec2, f64, Config, bool, f64, SimMode, usize) {
        (
            self.center,
            self.half_width,
            self.physics,
            self.sensitivity && self.mode == SimMode::FateMap,
            self.eta_b_ratio,
            self.mode,
            self.ecc_preset,
        )
    }

    pub fn scattering(&self) -> bool {
        self.mode == SimMode::Scattering
    }

    /// (v_inf / v_c, r_p / a) at the centre of scattering cell `c` (row 0 at the top).
    pub fn cell_params(c: UVec2) -> (f64, f64) {
        let n = SCATTER_CELLS as f64;
        (
            (c.x as f64 + 0.5) / n * SCATTER_V_MAX,
            (n - c.y as f64 - 0.5) / n * SCATTER_RP_MAX,
        )
    }

    /// World position of a texel coordinate (0..TEX_SIZE, y down).
    pub fn texel_to_world(&self, texel: Vec2) -> DVec2 {
        let half = TEX_SIZE as f64 / 2.0;
        let uv = texel.as_dvec2() - DVec2::splat(half);
        self.center + DVec2::new(uv.x, -uv.y) * self.scale()
    }

    /// The f32-rounded world position the GPU uses for simulated pixel `p` at `res`.
    pub fn gpu_pixel_world(&self, p: UVec2, res: u32) -> Vec2 {
        let tex = TEX_SIZE as f32;
        let block = tex / res as f32;
        let uv = (p.as_vec2() + 0.5) * block - 0.5 * tex;
        self.center.as_vec2() + Vec2::new(uv.x, -uv.y) * self.scale() as f32
    }
}

/// Bookkeeping for generations (view changes) and refinement levels.
#[derive(Resource, Debug)]
pub struct SimState {
    pub generation: u32,
    pub level: usize,
    pub front: u32,
    pub done: bool,
    pub steps_per_frame: u32,
    committed: Option<Settings>,
    prev_center: DVec2,
    prev_scale: f64,
    pub generation_started: f64,
    pub level_started: f64,
    pub finished_after: Option<f64>,
}

impl Default for SimState {
    fn default() -> Self {
        Self {
            generation: 0,
            level: 0,
            front: 0,
            done: false,
            steps_per_frame: 64,
            committed: None,
            prev_center: DVec2::ZERO,
            prev_scale: 1.0,
            generation_started: 0.0,
            level_started: 0.0,
            finished_after: None,
        }
    }
}

impl SimState {
    pub fn res(&self) -> u32 {
        LEVELS[self.level]
    }

    pub fn advance_level(&mut self, now: f64) {
        info!(
            "generation {} level {} ({}x{}) finished in {:.2}s",
            self.generation,
            self.level,
            self.res(),
            self.res(),
            now - self.level_started
        );
        if self.level + 1 < LEVELS.len() {
            self.level += 1;
            self.level_started = now;
        } else {
            self.done = true;
            self.finished_after = Some(now - self.generation_started);
        }
    }
}

/// Bumps the generation when anything that affects the simulation changed, then writes
/// the uniform for the render world.
pub fn commit_settings(
    mut settings: ResMut<Settings>,
    mut sim: ResMut<SimState>,
    mut uniform: ResMut<FractalUniform>,
    time: Res<Time<Real>>,
) {
    let min_hw = settings.min_half_width();
    if settings.half_width < min_hw {
        settings.half_width = min_hw;
    }
    settings.physics.masses = MASS_PRESETS[settings.mass_preset];

    let changed = match &sim.committed {
        None => true,
        Some(old) => old.sim_key() != settings.sim_key(),
    };
    let now = time.elapsed_secs_f64();
    if changed {
        let prev = sim.committed.as_ref().unwrap_or(&settings);
        (sim.prev_center, sim.prev_scale) = (prev.center, prev.scale());
        sim.generation = sim.generation.wrapping_add(1);
        sim.front ^= 1;
        // The scattering table runs one system per finest-level pixel from the start.
        sim.level = if settings.scattering() { LEVELS.len() - 1 } else { 0 };
        sim.done = false;
        sim.generation_started = now;
        sim.level_started = now;
        sim.finished_after = None;
        sim.committed = Some(settings.clone());
    } else if let Some(old) = &mut sim.committed {
        // Colour mode etc. don't need a re-run.
        old.color_mode = settings.color_mode;
    }

    let p = &settings.physics;
    let m = p.masses;
    uniform.0 = GpuParams {
        masses: Vec4::new(m[0] as f32, m[1] as f32, m[2] as f32, 0.0),
        center: settings.center.as_vec2(),
        prev_center: sim.prev_center.as_vec2(),
        pos_a: Vec2::new(p.pos_a[0] as f32, p.pos_a[1] as f32),
        pos_b: Vec2::new(p.pos_b[0] as f32, p.pos_b[1] as f32),
        scale: settings.scale() as f32,
        prev_scale: sim.prev_scale as f32,
        eta: p.eta as f32,
        eta_b: (p.eta * settings.eta_b_ratio) as f32,
        t_max_cr: p.t_max_cr as f32,
        r_far: p.r_far as f32,
        res: sim.res(),
        tex_size: TEX_SIZE,
        generation: sim.generation,
        level: sim.level as u32,
        steps_per_frame: sim.steps_per_frame,
        max_steps: p.max_steps,
        color_mode: match settings.color_mode {
            ColorMode::Fate => 0,
            ColorMode::EjectionTime => 1,
            ColorMode::EnergyError => 2,
            ColorMode::Steps => 3,
            ColorMode::Sensitivity => 4,
        },
        sensitivity: (settings.sensitivity && !settings.scattering()) as u32,
        front: sim.front,
        done: sim.done as u32,
        mode: settings.scattering() as u32,
        scatter_ecc: ECC_PRESETS[settings.ecc_preset] as f32,
        cells: SCATTER_CELLS,
        samples: SCATTER_SAMPLES,
        v_max: SCATTER_V_MAX as f32,
        rp_max: SCATTER_RP_MAX as f32,
        seed_base: sim.generation.wrapping_mul(0x9e37_79b9),
    };
}
