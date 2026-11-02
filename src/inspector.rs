//! Right-click to replay a system on the CPU (f64) and watch the orbits: the pixel's
//! three-body problem in the fate map, or one random encounter of a cell in the
//! scattering table.

use bevy::{
    math::DVec2,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use bevy_threebody::physics_cpu::{self, Encounter, Fate, Outcome, SCATTER_R0, V2};

use crate::input::DisplaySize;
use crate::params::{ECC_PRESETS, Settings};

#[derive(Message)]
pub enum InspectRequest {
    Close,
    /// Fate map: star 2 starting at this world position.
    Point(DVec2),
    /// Scattering table: a random encounter of this cell.
    Encounter(UVec2),
}

const MAX_POINTS: usize = 6000;
/// Seconds for one playback of the trajectory.
const PLAYBACK_SECS: f32 = 8.0;

pub struct Trajectory {
    pub label: String,
    /// Marker drawn at the start (fate map only).
    pub start: Option<DVec2>,
    pub outcome: Outcome,
    /// (time, positions) decimated to at most MAX_POINTS samples.
    pub points: Vec<(f64, [V2; 3])>,
    /// World centre and half-width to draw in.
    pub view: (DVec2, f64),
    pub scattering: bool,
}

#[derive(Resource, Default)]
pub struct Inspector {
    task: Option<Task<Trajectory>>,
    pub current: Option<Trajectory>,
    started: f32,
    replays: u64,
}

pub fn handle_requests(
    mut requests: MessageReader<InspectRequest>,
    mut inspector: ResMut<Inspector>,
    settings: Res<Settings>,
) {
    for request in requests.read() {
        let cfg = settings.physics;
        let task = match request {
            InspectRequest::Close => {
                inspector.task = None;
                inspector.current = None;
                continue;
            }
            InspectRequest::Point(c) => {
                let (c, view) = (*c, (settings.center, settings.half_width));
                AsyncComputeTaskPool::get().spawn(async move { trace_point(&cfg, c, view) })
            }
            InspectRequest::Encounter(cell) => {
                let (v_ratio, rp) = Settings::cell_params(*cell);
                let enc = Encounter { v_ratio, rp, ecc: ECC_PRESETS[settings.ecc_preset] };
                inspector.replays += 1;
                let seed = inspector.replays;
                AsyncComputeTaskPool::get().spawn(async move { trace_encounter(&cfg, enc, seed) })
            }
        };
        inspector.task = Some(task);
    }
}

/// Decimates by step (keeps detail in close encounters) and normalises time to 0..1.
fn decimate(first: [V2; 3], raw: &[(f64, [V2; 3])]) -> Vec<(f64, [V2; 3])> {
    let t_end = raw.last().map_or(1.0, |(t, _)| *t);
    let stride = raw.len().div_ceil(MAX_POINTS).max(1);
    let mut points = vec![(0.0, first)];
    points.extend(raw.iter().step_by(stride).map(|(t, x)| (t / t_end, *x)));
    if let Some((_, last)) = raw.last() {
        points.push((1.0, *last));
    }
    points
}

fn trace_point(cfg: &physics_cpu::Config, c: DVec2, view: (DVec2, f64)) -> Trajectory {
    let mut raw = Vec::new();
    let s = physics_cpu::initial_state(cfg, [c.x, c.y]);
    let outcome = physics_cpu::simulate_state(cfg, s, Some(&mut raw));
    let fate = match outcome.fate {
        Fate::Ejected(k) => format!("star {} ejected", ["gold", "blue", "red"][k]),
        _ => "undecided".into(),
    };
    Trajectory {
        label: format!(
            "inspect ({:.6}, {:.6}): {fate} after {:.1} t_cr",
            c.x, c.y, outcome.t_cr_units
        ),
        start: Some(c),
        outcome,
        points: decimate(physics_cpu::absolute_positions(&s, &cfg.masses), &raw),
        view,
        scattering: false,
    }
}

fn trace_encounter(cfg: &physics_cpu::Config, enc: Encounter, seed: u64) -> Trajectory {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5bd1_e995;
    let rand = std::array::from_fn(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    });
    let (s, sc, e_b0) = physics_cpu::scatter_state(cfg, &enc, rand);
    let mut raw = Vec::new();
    let outcome = physics_cpu::simulate_run(cfg, s, sc, Some(&mut raw));
    let (fate, binary_de) = match outcome.fate {
        Fate::Ejected(2) => ("flyby", Some((e_b0 - outcome.pair_energy) / e_b0.abs())),
        Fate::Ejected(k) => (
            ["exchange (gold binary star ejected)", "exchange (blue binary star ejected)"][k],
            Some((e_b0 - outcome.pair_energy) / e_b0.abs()),
        ),
        Fate::Ionized => ("ionisation", None),
        Fate::Undecided => ("unresolved", None),
    };
    Trajectory {
        label: format!(
            "replay v/vc {:.2}, rp/a {:.2}: {fate} after {:.0} binary orbits{}",
            enc.v_ratio,
            enc.rp,
            outcome.t_cr_units,
            binary_de.map_or(String::new(), |d| format!(", binary dE/E {d:+.3}"))
        ),
        start: None,
        outcome,
        points: decimate(physics_cpu::absolute_positions(&s, &cfg.masses), &raw),
        view: (DVec2::ZERO, 1.25 * SCATTER_R0),
        scattering: true,
    }
}

pub fn poll_task(mut inspector: ResMut<Inspector>, time: Res<Time<Real>>) {
    let Some(task) = inspector.task.as_mut() else { return };
    if let Some(traj) = block_on(poll_once(task)) {
        inspector.current = Some(traj);
        inspector.task = None;
        inspector.started = time.elapsed_secs();
    }
}

pub const STAR_COLORS: [Color; 3] = [
    Color::srgb(1.0, 0.77, 0.20),
    Color::srgb(0.22, 0.52, 1.0),
    Color::srgb(0.93, 0.22, 0.24),
];

/// World position to 2D camera coordinates (logical pixels, origin at window centre).
fn to_screen(view: (DVec2, f64), display: f32, p: DVec2) -> Vec2 {
    ((p - view.0) / view.1).as_vec2() * (display / 2.0)
}

pub fn draw(
    mut gizmos: Gizmos,
    inspector: Res<Inspector>,
    settings: Res<Settings>,
    display: Res<DisplaySize>,
    time: Res<Time<Real>>,
) {
    let half = display.0 / 2.0;
    if settings.scattering() {
        // Hard/soft boundary: v_inf = v_c.
        let x = (1.0 / crate::params::SCATTER_V_MAX as f32 - 0.5) * display.0;
        gizmos.line_2d(Vec2::new(x, -half), Vec2::new(x, half), Color::srgba(1.0, 1.0, 1.0, 0.6));
    } else {
        // Fixed starting points of stars 0 and 1.
        let view = (settings.center, settings.half_width);
        let ph = &settings.physics;
        for (k, p) in [ph.pos_a, ph.pos_b].into_iter().enumerate() {
            let p = to_screen(view, display.0, DVec2::new(p[0], p[1]));
            gizmos.circle_2d(p, 4.0, STAR_COLORS[k].with_alpha(0.9));
        }
    }

    let Some(traj) = &inspector.current else { return };
    if traj.scattering != settings.scattering() {
        return;
    }
    let s = |p: V2| to_screen(traj.view, display.0, DVec2::new(p[0], p[1]));
    if let Some(c) = traj.start {
        gizmos.circle_2d(s([c.x, c.y]), 5.0, Color::WHITE);
    }
    let elapsed = time.elapsed_secs() - inspector.started;
    let phase = (elapsed / PLAYBACK_SECS).fract() as f64;
    let shown = traj.points.partition_point(|(t, _)| *t <= phase).max(2);
    for k in 0..3 {
        let pts = traj.points[..shown].iter().map(|(_, x)| s(x[k]));
        gizmos.linestrip_2d(pts, STAR_COLORS[k].with_alpha(0.85));
        let (_, now) = traj.points[shown - 1];
        gizmos.circle_2d(s(now[k]), 6.0, STAR_COLORS[k]);
    }
}

pub fn describe(traj: &Trajectory) -> String {
    format!(
        "{}, {} steps, energy error {:.1e}",
        traj.label, traj.outcome.steps, traj.outcome.energy_err
    )
}
