//! Reads GPU stats back every frame, drives refinement levels, tunes the per-frame step
//! budget, and checks GPU fates against the CPU reference on request.

use bevy::{
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete, ReadbackOnce},
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use bevy_threebody::physics_cpu::{self, Fate};

use crate::params::{LEVELS, Settings, SimState};
use crate::render::FractalTargets;

pub const STATS_HEADER: usize = 16;
pub const GRID: usize = 64;
pub const STATS_LEN: usize = STATS_HEADER + GRID * GRID * 4;

/// Latest stats for the current generation and level.
#[derive(Resource, Default, Debug, Clone)]
pub struct GpuStats {
    pub generation: u32,
    pub level: u32,
    /// fate 0, fate 1, fate 2, undecided
    pub counts: [u32; 4],
    pub running: u32,
    pub disagree: u32,
    /// Per 64x64 cell: [fate0, fate1, fate2, undecided].
    pub grid: Vec<[u32; 4]>,
    pub valid: bool,
}

impl GpuStats {
    pub fn finished(&self) -> u32 {
        self.counts.iter().sum()
    }
}

pub fn spawn_readback(mut commands: Commands, targets: Res<FractalTargets>) {
    commands.spawn(Readback::buffer(targets.stats.clone())).observe(
        |event: On<ReadbackComplete>, sim: Res<SimState>, mut stats: ResMut<GpuStats>| {
            let data: Vec<u32> = event.to_shader_type();
            if data.len() < STATS_LEN
                || data[0] != sim.generation
                || data[1] != sim.level as u32
            {
                return;
            }
            stats.generation = data[0];
            stats.level = data[1];
            stats.counts = [data[2], data[3], data[4], data[5]];
            stats.running = data[6];
            stats.disagree = data[7];
            stats.grid = data[STATS_HEADER..STATS_LEN]
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect();
            stats.valid = true;
        },
    );
}

/// Moves to the next refinement level once every pixel of the current one is finished.
pub fn advance_levels(
    mut sim: ResMut<SimState>,
    mut stats: ResMut<GpuStats>,
    settings: Res<Settings>,
    time: Res<Time<Real>>,
) {
    if settings.scattering() {
        return;
    }
    if stats.valid && (stats.generation != sim.generation || stats.level != sim.level as u32) {
        stats.valid = false;
    }
    if sim.done || !stats.valid {
        return;
    }
    let total = LEVELS[sim.level] * LEVELS[sim.level];
    if stats.running == 0 && stats.finished() >= total {
        sim.advance_level(time.elapsed_secs_f64());
        stats.valid = false;
    }
}

/// Grows or shrinks the per-frame step budget to keep the frame rate interactive.
pub fn tune_step_budget(mut sim: ResMut<SimState>, time: Res<Time<Real>>) {
    let dt = time.delta_secs();
    if dt <= 0.0 || sim.done {
        return;
    }
    let budget = sim.steps_per_frame as f32;
    let next = if dt < 0.019 {
        budget * 1.1
    } else if dt > 0.030 {
        budget * 0.75
    } else {
        budget
    };
    // Coarse levels have fewer pixels, so they can afford more steps per pixel.
    let max = 4096 * (LEVELS[LEVELS.len() - 1] / LEVELS[sim.level]).pow(2);
    sim.steps_per_frame = (next.round() as u32).clamp(8, max);
}

// ---------------------------------------------------------------------------------------
// GPU vs CPU verification

#[derive(Resource, Default)]
pub struct Verify {
    pub pending: bool,
    task: Option<Task<VerifyReport>>,
    pub report: Option<VerifyReport>,
}

#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub samples: usize,
    pub early_agree: usize,
    pub early_total: usize,
    pub late_agree: usize,
    pub late_total: usize,
}

impl VerifyReport {
    pub fn summary(&self) -> String {
        let pct = |a: usize, n: usize| if n == 0 { 0.0 } else { 100.0 * a as f64 / n as f64 };
        format!(
            "GPU vs CPU on {} pixels: early ejections (<10 t_cr) agree {:.1}% ({}), later {:.1}% ({})",
            self.samples,
            pct(self.early_agree, self.early_total),
            self.early_total,
            pct(self.late_agree, self.late_total),
            self.late_total
        )
    }
}

const VERIFY_SAMPLES: usize = 400;

/// Reads the per-pixel fates back once and compares random pixels with the CPU.
pub fn start_verify(
    mut commands: Commands,
    mut verify: ResMut<Verify>,
    targets: Res<FractalTargets>,
    sim: Res<SimState>,
) {
    if !verify.pending || verify.task.is_some() || !sim.done {
        return;
    }
    verify.pending = false;
    let res = sim.res();
    commands.spawn(ReadbackOnce::buffer(targets.fates.clone())).observe(
        move |event: On<ReadbackComplete>, settings: Res<Settings>, mut verify: ResMut<Verify>| {
            let info: Vec<u32> = event.to_shader_type();
            let settings = settings.clone();
            verify.task = Some(AsyncComputeTaskPool::get().spawn(async move {
                compare_with_cpu(&settings, res, &info)
            }));
        },
    );
}

pub fn finish_verify(mut verify: ResMut<Verify>) {
    let Some(task) = verify.task.as_mut() else { return };
    if let Some(report) = block_on(poll_once(task)) {
        info!("{}", report.summary());
        verify.report = Some(report);
        verify.task = None;
    }
}

fn compare_with_cpu(settings: &Settings, res: u32, info: &[u32]) -> VerifyReport {
    let mut report = VerifyReport {
        samples: 0,
        early_agree: 0,
        early_total: 0,
        late_agree: 0,
        late_total: 0,
    };
    // Deterministic LCG so runs are comparable.
    let mut seed = 0x2545_f491_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for _ in 0..VERIFY_SAMPLES {
        let p = UVec2::new(next() % res, next() % res);
        let word = info[(p.y * res + p.x) as usize];
        let gpu = match word & 3 {
            1 => Fate::Ejected(((word >> 2) & 3) as usize),
            2 => Fate::Undecided,
            _ => continue,
        };
        let c = settings.gpu_pixel_world(p, res);
        let cpu = physics_cpu::simulate(&settings.physics, [c.x as f64, c.y as f64]);
        report.samples += 1;
        let agree = cpu.fate == gpu;
        if matches!(cpu.fate, Fate::Ejected(_)) && cpu.t_cr_units < 10.0 {
            report.early_total += 1;
            report.early_agree += agree as usize;
        } else {
            report.late_total += 1;
            report.late_agree += agree as usize;
        }
    }
    report
}
