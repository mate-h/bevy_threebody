//! Text overlay: view, progress, fate percentages, inspector and verification results.

use bevy::prelude::*;

use crate::inspector::{Inspector, describe};
use crate::params::{LEVELS, Settings, SimState};
use crate::stats::{GpuStats, Verify};

#[derive(Component)]
pub struct StatusText;

#[derive(Component)]
pub struct HelpText;

#[derive(Resource)]
pub struct ShowHelp(pub bool);

const HELP: &str = "\
wheel zoom  |  drag pan  |  right-click inspect orbit  |  esc close
space zoom 5x into boundary  |  backspace zoom out 5x  |  T auto tour (to 625x)  |  R reset
1 fate  |  2 ejection time  |  3 energy error  |  4 steps  |  5 sensitivity  |  S toggle 2nd run
[ ] eta  |  - = t_max  |  M masses  |  V verify vs CPU  |  P screenshot  |  H hide help";

pub fn setup_ui(mut commands: Commands) {
    let font = TextFont { font_size: FontSize::Px(13.0), ..default() };
    commands.spawn((
        StatusText,
        Text::new(""),
        font.clone(),
        TextColor(Color::srgb(0.92, 0.92, 0.92)),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            top: px(8),
            left: px(10),
            padding: UiRect::all(px(6)),
            ..default()
        },
    ));
    commands.spawn((
        HelpText,
        Text::new(HELP),
        font,
        TextColor(Color::srgba(0.85, 0.85, 0.85, 0.9)),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(8),
            left: px(10),
            padding: UiRect::all(px(6)),
            ..default()
        },
    ));
}

pub fn update_ui(
    settings: Res<Settings>,
    sim: Res<SimState>,
    stats: Res<GpuStats>,
    verify: Res<Verify>,
    inspector: Res<Inspector>,
    help: Res<ShowHelp>,
    time: Res<Time<Real>>,
    mut status: Single<&mut Text, (With<StatusText>, Without<HelpText>)>,
    mut help_vis: Single<&mut Visibility, With<HelpText>>,
) {
    **help_vis = if help.0 { Visibility::Inherited } else { Visibility::Hidden };

    let p = &settings.physics;
    let mut lines = vec![
        format!(
            "zoom {:.1}x  centre ({:.8}, {:.8})  masses {:?}",
            settings.zoom_factor(),
            settings.center.x,
            settings.center.y,
            p.masses
        ),
        format!(
            "eta {:.4}  t_max {:.0} t_cr  r_far {:.0} r_v  colour: {}{}",
            p.eta,
            p.t_max_cr,
            p.r_far,
            settings.color_mode.label(),
            if settings.sensitivity { "  (2nd run at eta x0.5)" } else { "" }
        ),
    ];
    let res = LEVELS[sim.level];
    let elapsed = sim
        .finished_after
        .unwrap_or(time.elapsed_secs_f64() - sim.generation_started);
    let progress = if stats.valid {
        let total = (res * res) as f64;
        100.0 * (total - stats.running as f64) / total
    } else {
        0.0
    };
    lines.push(if sim.done {
        format!("done {res}x{res} in {elapsed:.1}s")
    } else {
        format!(
            "level {}/{} ({res}x{res}) {progress:.1}%  {elapsed:.1}s  {} steps/frame",
            sim.level + 1,
            LEVELS.len(),
            sim.steps_per_frame
        )
    });
    if stats.valid {
        let n = (stats.finished() + stats.running).max(1) as f64;
        let pct = |k: u32| 100.0 * k as f64 / n;
        let mut s = format!(
            "gold {:.1}%  blue {:.1}%  red {:.1}%  undecided {:.1}%",
            pct(stats.counts[0]),
            pct(stats.counts[1]),
            pct(stats.counts[2]),
            pct(stats.counts[3])
        );
        if settings.sensitivity {
            s += &format!("  disagree {:.1}%", pct(stats.disagree));
        }
        lines.push(s);
    }
    if settings.half_width <= settings.min_half_width() * 1.0001 {
        lines.push("f32 precision limit: neighbouring pixels can't be told apart beyond this zoom".into());
    }
    if let Some(traj) = &inspector.current {
        lines.push(describe(traj));
    }
    if let Some(report) = &verify.report {
        lines.push(report.summary());
    }
    status.0 = lines.join("\n");
}
