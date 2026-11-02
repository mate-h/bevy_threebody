//! Three-body ejection fractal: every pixel is a three-body problem, coloured by which
//! star gets thrown out. See PLAN.md.

mod input;
mod inspector;
mod params;
mod render;
mod scatter;
mod stats;
mod ui;

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{render_resource::*, storage::ShaderBuffer},
    render::view::screenshot::{Screenshot, save_to_disk},
    window::WindowResolution,
};

use params::{FractalUniform, Settings, SimState, TEX_SIZE};
use render::{FractalComputePlugin, FractalTargets};

fn main() {
    let autopilot = Autopilot::from_args();
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Three-body ejection fractal".into(),
                    resolution: WindowResolution::new(900, 900),
                    ..default()
                }),
                ..default()
            }),
            FractalComputePlugin,
        ))
        .add_message::<inspector::InspectRequest>()
        .insert_resource(autopilot.initial_settings())
        .insert_resource(autopilot)
        .insert_resource(FractalUniform(Default::default()))
        .insert_resource(ui::ShowHelp(true))
        .init_resource::<SimState>()
        .init_resource::<stats::GpuStats>()
        .init_resource::<stats::Verify>()
        .init_resource::<scatter::ScatterTable>()
        .init_resource::<scatter::ScatterVerify>()
        .init_resource::<input::DisplaySize>()
        .init_resource::<input::DragState>()
        .init_resource::<input::AutoZoom>()
        .init_resource::<inspector::Inspector>()
        .add_systems(
            Startup,
            (setup, ui::setup_ui, stats::spawn_readback, scatter::spawn_readback).chain(),
        )
        .add_systems(
            Update,
            (
                input::fit_sprite,
                (input::mouse_controls, input::keyboard_controls),
                (input::auto_zoom_pause, input::run_auto_zoom).chain(),
                run_autopilot,
                params::commit_settings,
                stats::advance_levels,
                stats::tune_step_budget,
                (stats::start_verify, stats::finish_verify),
                (scatter::track_rate, scatter::start_verify, scatter::finish_verify),
                (inspector::handle_requests, inspector::poll_task, inspector::draw).chain(),
                show_front_image,
                ui::update_ui,
            )
                .chain(),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let mut image = Image::new_target_texture(TEX_SIZE, TEX_SIZE, render::OUTPUT_FORMAT, None);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let images = [images.add(image.clone()), images.add(image)];

    let buffer = |len: usize| {
        let mut b = ShaderBuffer::with_size(4 * len as u64, RenderAssetUsages::RENDER_WORLD);
        b.buffer_usage |= BufferUsages::COPY_SRC;
        b
    };
    let stats = buffers.add(buffer(stats::STATS_LEN));
    let fates = buffers.add(buffer((TEX_SIZE * TEX_SIZE) as usize));

    commands.spawn(Camera2d);
    commands.spawn((
        input::FractalSprite,
        Sprite {
            image: images[0].clone(),
            custom_size: Some(Vec2::splat(900.0)),
            ..default()
        },
    ));
    commands.insert_resource(FractalTargets { images, stats, fates });
}

fn show_front_image(
    targets: Res<FractalTargets>,
    sim: Res<SimState>,
    mut sprite: Single<&mut Sprite, With<input::FractalSprite>>,
) {
    let front = &targets.images[sim.front as usize];
    if sprite.image != *front {
        sprite.image = front.clone();
    }
}

/// Command-line automation for unattended runs:
///   --mode N (1-5)  --masses N (0-2)  --sensitivity  --tour
///   --scatter       start in the scattering table   --ecc N (0-3) its eccentricity preset
///   --shot-after S  screenshot (and verify) after S seconds instead of on completion
///   --shot PREFIX   save a screenshot whenever an image completes
///   --verify        compare GPU fates against the CPU when an image completes
///   --exit          quit once everything requested is done
#[derive(Resource, Default)]
struct Autopilot {
    mode: Option<u32>,
    masses: Option<usize>,
    sensitivity: bool,
    tour: bool,
    shot: Option<String>,
    verify: bool,
    exit: bool,
    scatter: bool,
    ecc: Option<usize>,
    shot_after: Option<f64>,
    timed_done: bool,
    handled_generation: Option<u32>,
    tour_started: bool,
    exit_at: Option<f64>,
}

impl Autopilot {
    fn from_args() -> Self {
        let mut a = Autopilot::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => a.mode = args.next().and_then(|v| v.parse().ok()),
                "--masses" => a.masses = args.next().and_then(|v| v.parse().ok()),
                "--sensitivity" => a.sensitivity = true,
                "--tour" => a.tour = true,
                "--shot" => a.shot = args.next(),
                "--verify" => a.verify = true,
                "--exit" => a.exit = true,
                "--scatter" => a.scatter = true,
                "--ecc" => a.ecc = args.next().and_then(|v| v.parse().ok()),
                "--shot-after" => a.shot_after = args.next().and_then(|v| v.parse().ok()),
                other => warn!("unknown argument {other}"),
            }
        }
        a
    }

    fn initial_settings(&self) -> Settings {
        use params::ColorMode::*;
        let mut s = Settings::default();
        if let Some(m) = self.masses {
            s.mass_preset = m.min(params::MASS_PRESETS.len() - 1);
        }
        s.sensitivity = self.sensitivity;
        if self.scatter {
            s.mode = params::SimMode::Scattering;
        }
        if let Some(e) = self.ecc {
            s.ecc_preset = e.min(params::ECC_PRESETS.len() - 1);
        }
        s.color_mode = match self.mode {
            Some(2) => EjectionTime,
            Some(3) => EnergyError,
            Some(4) => Steps,
            Some(5) => {
                s.sensitivity = true;
                Sensitivity
            }
            _ => Fate,
        };
        s
    }
}

fn run_autopilot(
    mut autopilot: ResMut<Autopilot>,
    mut commands: Commands,
    sim: Res<SimState>,
    settings: Res<Settings>,
    mut verify: ResMut<stats::Verify>,
    mut scatter_verify: ResMut<scatter::ScatterVerify>,
    mut auto: ResMut<input::AutoZoom>,
    time: Res<Time<Real>>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs_f64();
    if autopilot.tour && !autopilot.tour_started {
        autopilot.tour_started = true;
        *auto = input::AutoZoom {
            active: true,
            steps_left: input::TOUR_STEPS,
            wait_until: f64::INFINITY,
        };
    }
    if let Some(after) = autopilot.shot_after {
        if !autopilot.timed_done && now >= after {
            autopilot.timed_done = true;
            if let Some(prefix) = &autopilot.shot {
                let path = format!("{prefix}-{after:.0}s.png");
                info!("saving screenshot {path}");
                commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
            }
            if autopilot.verify {
                scatter_verify.pending = settings.scattering();
                verify.pending = !settings.scattering();
            }
        }
        let verified = !autopilot.verify || scatter_verify.report.is_some() || verify.report.is_some();
        if autopilot.exit && autopilot.timed_done && verified {
            let at = *autopilot.exit_at.get_or_insert(now + 1.5);
            if now >= at {
                exit.write(AppExit::Success);
            }
        }
        return;
    }
    if sim.done && autopilot.handled_generation != Some(sim.generation) {
        autopilot.handled_generation = Some(sim.generation);
        if let Some(prefix) = &autopilot.shot {
            let path = format!("{prefix}-{:.0}x.png", settings.zoom_factor());
            info!("saving screenshot {path}");
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
        }
        if autopilot.verify {
            verify.pending = true;
        }
    }
    let finished = sim.done
        && autopilot.handled_generation == Some(sim.generation)
        && !(autopilot.tour && auto.active)
        && (!autopilot.verify || verify.report.is_some());
    if autopilot.exit && finished {
        let at = *autopilot.exit_at.get_or_insert(now + 1.5);
        if now >= at {
            exit.write(AppExit::Success);
        }
    }
}
