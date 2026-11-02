//! Scattering-table mode: reads the per-cell outcome counters back, tracks the encounter
//! rate, and checks GPU probabilities against the CPU reference.

use bevy::{
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use bevy_threebody::physics_cpu::{Encounter, Fate, simulate_encounter};

use crate::params::{ECC_PRESETS, SCATTER_CELLS, Settings, SimState};
use crate::render::FractalTargets;

/// Counters per cell; matches ACC_* in the shader.
const ACC_STRIDE: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
pub struct CellCounts {
    pub trials: u32,
    pub flyby: u32,
    pub exchange: u32,
    pub ionized: u32,
    pub undecided: u32,
    de_sum: i32,
    de_count: u32,
}

impl CellCounts {
    fn from_words(w: &[u32]) -> Self {
        Self {
            trials: w[0],
            flyby: w[1],
            exchange: w[2],
            ionized: w[3],
            undecided: w[4],
            de_sum: w[5] as i32,
            de_count: w[6],
        }
    }

    pub fn add(&mut self, o: &CellCounts) {
        self.trials += o.trials;
        self.flyby += o.flyby;
        self.exchange += o.exchange;
        self.ionized += o.ionized;
        self.undecided += o.undecided;
        self.de_sum = self.de_sum.wrapping_add(o.de_sum);
        self.de_count += o.de_count;
    }

    /// Percentages of (flyby, exchange, ionisation, unresolved).
    pub fn percentages(&self) -> [f64; 4] {
        let n = self.trials.max(1) as f64;
        [self.flyby, self.exchange, self.ionized, self.undecided].map(|k| 100.0 * k as f64 / n)
    }

    /// Mean fractional change of the binary's binding energy (positive = hardened).
    pub fn mean_de(&self) -> f64 {
        self.de_sum as f64 / 1000.0 / self.de_count.max(1) as f64
    }

    pub fn describe(&self) -> String {
        let [f, e, i, u] = self.percentages();
        format!(
            "n {}  flyby {f:.1}%  exchange {e:.1}%  ionised {i:.1}%  unresolved {u:.1}%  binary dE/E {:+.3}",
            self.trials,
            self.mean_de()
        )
    }
}

#[derive(Resource, Default)]
pub struct ScatterTable {
    /// Row-major, row 0 at the top (largest r_p).
    pub cells: Vec<CellCounts>,
    pub total: u64,
    pub rate: f64,
    last: Option<(f64, u64)>,
}

impl ScatterTable {
    pub fn cell(&self, c: UVec2) -> Option<&CellCounts> {
        self.cells.get((c.y * SCATTER_CELLS + c.x) as usize)
    }

    /// All cells at one speed (one column), summed over r_p.
    pub fn column(&self, x: u32) -> CellCounts {
        let mut sum = CellCounts::default();
        for y in 0..SCATTER_CELLS {
            if let Some(c) = self.cell(UVec2::new(x, y)) {
                sum.add(c);
            }
        }
        sum
    }
}

pub fn spawn_readback(mut commands: Commands, targets: Res<FractalTargets>) {
    let bytes = (SCATTER_CELLS * SCATTER_CELLS) as u64 * ACC_STRIDE as u64 * 4;
    commands.spawn(Readback::buffer_range(targets.fates.clone(), 0, bytes)).observe(
        |event: On<ReadbackComplete>,
         settings: Res<Settings>,
         sim: Res<SimState>,
         mut table: ResMut<ScatterTable>| {
            if !settings.scattering() {
                return;
            }
            let data: Vec<u32> = event.to_shader_type();
            if data.len() < 8 || data[7] != sim.generation {
                return;
            }
            table.cells = data.chunks_exact(ACC_STRIDE).map(CellCounts::from_words).collect();
            table.total = table.cells.iter().map(|c| c.trials as u64).sum();
        },
    );
}

/// Encounters per second, smoothed.
pub fn track_rate(mut table: ResMut<ScatterTable>, time: Res<Time<Real>>) {
    let now = time.elapsed_secs_f64();
    let total = table.total;
    match table.last {
        Some((t, n)) if now - t >= 1.0 => {
            let r = total.saturating_sub(n) as f64 / (now - t);
            table.rate = if table.rate == 0.0 { r } else { 0.5 * table.rate + 0.5 * r };
            table.last = Some((now, total));
        }
        None => table.last = Some((now, total)),
        _ => {}
    }
    if total < table.last.map_or(0, |l| l.1) {
        // Table was reset.
        table.last = Some((now, total));
        table.rate = 0.0;
    }
}

// ---------------------------------------------------------------------------------------
// GPU vs CPU

#[derive(Resource, Default)]
pub struct ScatterVerify {
    pub pending: bool,
    task: Option<Task<String>>,
    pub report: Option<String>,
}

const VERIFY_TRIALS: usize = 300;
const VERIFY_POINTS: [(f64, f64); 6] = [(0.25, 0.5), (0.25, 2.0), (0.75, 0.5), (1.5, 0.5), (1.5, 2.0), (2.5, 0.5)];

fn cell_at(v: f64, rp: f64) -> UVec2 {
    let n = SCATTER_CELLS as f64;
    let x = (v / crate::params::SCATTER_V_MAX * n).floor().clamp(0.0, n - 1.0);
    let y = (n - rp / crate::params::SCATTER_RP_MAX * n).floor().clamp(0.0, n - 1.0);
    UVec2::new(x as u32, y as u32)
}

pub fn start_verify(mut verify: ResMut<ScatterVerify>, settings: Res<Settings>, table: Res<ScatterTable>) {
    if !verify.pending || verify.task.is_some() || !settings.scattering() || table.cells.is_empty() {
        return;
    }
    verify.pending = false;
    let mut cfg = settings.physics;
    cfg.masses = crate::params::MASS_PRESETS[settings.mass_preset];
    let ecc = ECC_PRESETS[settings.ecc_preset];
    let gpu: Vec<(UVec2, CellCounts)> = VERIFY_POINTS
        .iter()
        .map(|&(v, rp)| {
            let c = cell_at(v, rp);
            (c, *table.cell(c).unwrap())
        })
        .collect();
    verify.task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let mut lines = vec![format!(
            "GPU vs CPU ({VERIFY_TRIALS} CPU f64 trials per cell; 2 sigma ~ {:.0}%)",
            200.0 * (0.25 / VERIFY_TRIALS as f64).sqrt()
        )];
        let mut worst: f64 = 0.0;
        for (c, g) in gpu {
            let (v_ratio, rp) = Settings::cell_params(c);
            let enc = Encounter { v_ratio, rp, ecc };
            let outs: Vec<Fate> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..VERIFY_TRIALS)
                    .map(|n| {
                        sc.spawn(move || {
                            let mut seed = (n as u64 + 17).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                            let r = std::array::from_fn(|_| {
                                seed = seed
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                (seed >> 11) as f64 / (1u64 << 53) as f64
                            });
                            simulate_encounter(&cfg, &enc, r).0.fate
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let pct = |f: &dyn Fn(&Fate) -> bool| 100.0 * outs.iter().filter(|o| f(o)).count() as f64 / outs.len() as f64;
            let cpu = [
                pct(&|f| *f == Fate::Ejected(2)),
                pct(&|f| matches!(f, Fate::Ejected(0) | Fate::Ejected(1))),
                pct(&|f| *f == Fate::Ionized),
            ];
            let gp = g.percentages();
            for k in 0..3 {
                worst = worst.max((cpu[k] - gp[k]).abs());
            }
            lines.push(format!(
                "  v/vc {v_ratio:.2} rp {rp:.2}: flyby {:.0}/{:.0}  exchange {:.0}/{:.0}  ionised {:.0}/{:.0} (GPU/CPU %, GPU n {})",
                gp[0], cpu[0], gp[1], cpu[1], gp[2], cpu[2], g.trials
            ));
        }
        lines.push(format!("  largest difference {worst:.1} points"));
        lines.join("\n")
    }));
}

pub fn finish_verify(mut verify: ResMut<ScatterVerify>) {
    let Some(task) = verify.task.as_mut() else { return };
    if let Some(report) = block_on(poll_once(task)) {
        info!("{report}");
        verify.report = Some(report);
        verify.task = None;
    }
}
