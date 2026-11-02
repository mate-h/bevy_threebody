//! Right-click a pixel to replay its three-body system on the CPU (f64) and watch the
//! orbits drawn over the fractal.

use bevy::{
    math::DVec2,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use bevy_threebody::physics_cpu::{self, Fate, Outcome, V2};

use crate::input::DisplaySize;
use crate::params::Settings;

/// `Some(world position)` to inspect that starting point, `None` to close.
#[derive(Message)]
pub struct InspectRequest(pub Option<DVec2>);

const MAX_POINTS: usize = 6000;
/// Seconds for one playback of the trajectory.
const PLAYBACK_SECS: f32 = 8.0;

pub struct Trajectory {
    pub start: DVec2,
    pub outcome: Outcome,
    /// (time, positions) decimated to at most MAX_POINTS samples.
    pub points: Vec<(f64, [V2; 3])>,
}

#[derive(Resource, Default)]
pub struct Inspector {
    task: Option<Task<Trajectory>>,
    pub current: Option<Trajectory>,
    started: f32,
}

pub fn handle_requests(
    mut requests: MessageReader<InspectRequest>,
    mut inspector: ResMut<Inspector>,
    settings: Res<Settings>,
) {
    for InspectRequest(target) in requests.read() {
        match target {
            None => {
                inspector.task = None;
                inspector.current = None;
            }
            Some(c) => {
                let cfg = settings.physics;
                let c = *c;
                inspector.task = Some(AsyncComputeTaskPool::get().spawn(async move {
                    trace(&cfg, c)
                }));
            }
        }
    }
}

fn trace(cfg: &physics_cpu::Config, c: DVec2) -> Trajectory {
    let mut raw = Vec::new();
    let s = physics_cpu::initial_state(cfg, [c.x, c.y]);
    let outcome = physics_cpu::simulate_state(cfg, s, Some(&mut raw));
    let t_end = raw.last().map_or(1.0, |(t, _)| *t);
    // Decimate by step (keeps detail in close encounters); play back in real time.
    let stride = raw.len().div_ceil(MAX_POINTS).max(1);
    let mut points = vec![(0.0, physics_cpu::absolute_positions(&s, &cfg.masses))];
    points.extend(raw.iter().step_by(stride).map(|(t, x)| (t / t_end, *x)));
    if let Some((_, last)) = raw.last() {
        points.push((1.0, *last));
    }
    Trajectory { start: c, outcome, points }
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
fn to_screen(settings: &Settings, display: f32, p: DVec2) -> Vec2 {
    ((p - settings.center) / settings.half_width).as_vec2() * (display / 2.0)
}

pub fn draw(
    mut gizmos: Gizmos,
    inspector: Res<Inspector>,
    settings: Res<Settings>,
    display: Res<DisplaySize>,
    time: Res<Time<Real>>,
) {
    let s = |p: V2| to_screen(&settings, display.0, DVec2::new(p[0], p[1]));
    // Fixed starting points of stars 0 and 1.
    let ph = &settings.physics;
    for (k, p) in [ph.pos_a, ph.pos_b].into_iter().enumerate() {
        gizmos.circle_2d(s(p), 4.0, STAR_COLORS[k].with_alpha(0.9));
    }

    let Some(traj) = &inspector.current else { return };
    gizmos.circle_2d(s([traj.start.x, traj.start.y]), 5.0, Color::WHITE);
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
    let fate = match traj.outcome.fate {
        Fate::Ejected(k) => format!(
            "star {} ejected after {:.1} t_cr",
            ["gold", "blue", "red"][k],
            traj.outcome.t_cr_units
        ),
        Fate::Undecided => format!("undecided after {:.1} t_cr", traj.outcome.t_cr_units),
    };
    format!(
        "inspect ({:.6}, {:.6}): {fate}, {} steps, dE/E {:.1e}",
        traj.start.x, traj.start.y, traj.outcome.steps, traj.outcome.energy_err
    )
}
