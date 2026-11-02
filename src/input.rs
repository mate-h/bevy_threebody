//! Mouse and keyboard controls, auto-zoom, screenshots.

use bevy::{
    input::mouse::{AccumulatedMouseScroll, MouseScrollUnit},
    math::DVec2,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
    window::PrimaryWindow,
};

use crate::params::{ColorMode, DEFAULT_HALF_WIDTH, MASS_PRESETS, Settings, SimState, TEX_SIZE};
use crate::stats::{GRID, GpuStats, Verify};
use crate::inspector::InspectRequest;

/// The sprite showing the fractal; sized to fit the window.
#[derive(Component)]
pub struct FractalSprite;

/// Side length of the displayed square, in logical pixels.
#[derive(Resource, Default)]
pub struct DisplaySize(pub f32);

#[derive(Resource, Default)]
pub struct DragState {
    last: Option<Vec2>,
}

/// Auto-zoom tour: repeatedly zoom 5x into the most mixed boundary region.
#[derive(Resource)]
pub struct AutoZoom {
    pub active: bool,
    pub steps_left: u32,
    pub wait_until: f64,
}

impl Default for AutoZoom {
    fn default() -> Self {
        Self { active: false, steps_left: 0, wait_until: 0.0 }
    }
}

pub const ZOOM_STEP: f64 = 5.0;
pub const TOUR_STEPS: u32 = 4; // 5x four times: 5, 25, 125, 625

pub fn fit_sprite(
    window: Single<&Window, With<PrimaryWindow>>,
    mut sprite: Single<&mut Sprite, With<FractalSprite>>,
    mut size: ResMut<DisplaySize>,
) {
    let side = window.width().min(window.height());
    if (size.0 - side).abs() > 0.5 {
        size.0 = side;
        sprite.custom_size = Some(Vec2::splat(side));
    }
}

/// Cursor position as a texel coordinate (0..TEX_SIZE, y down), if over the fractal.
pub fn cursor_texel(window: &Window, display: f32) -> Option<Vec2> {
    let cursor = window.cursor_position()?;
    let center = Vec2::new(window.width(), window.height()) / 2.0;
    let rel = (cursor - center) / display + 0.5;
    (rel.cmpge(Vec2::ZERO).all() && rel.cmple(Vec2::ONE).all()).then(|| rel * TEX_SIZE as f32)
}

pub fn mouse_controls(
    window: Single<&Window, With<PrimaryWindow>>,
    display: Res<DisplaySize>,
    scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    mut settings: ResMut<Settings>,
    mut auto: ResMut<AutoZoom>,
    mut inspect: MessageWriter<InspectRequest>,
) {
    let texel = cursor_texel(&window, display.0);

    // Wheel: zoom about the cursor.
    let lines = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    if lines != 0.0 {
        if let Some(texel) = texel {
            let anchor = settings.texel_to_world(texel);
            let factor = 1.2f64.powf(-lines as f64);
            let old_hw = settings.half_width;
            settings.half_width = (old_hw * factor).min(DEFAULT_HALF_WIDTH * 4.0);
            let k = settings.half_width / old_hw;
            settings.center = anchor + (settings.center - anchor) * k;
            auto.active = false;
        }
    }

    // Left drag: pan.
    if buttons.pressed(MouseButton::Left) {
        if let Some(pos) = window.cursor_position() {
            if let Some(last) = drag.last {
                let delta = (pos - last) / display.0 * TEX_SIZE as f32;
                if delta != Vec2::ZERO {
                    let scale = settings.scale();
                    settings.center -= DVec2::new(delta.x as f64, -delta.y as f64) * scale;
                    auto.active = false;
                }
            }
            drag.last = Some(pos);
        }
    } else {
        drag.last = None;
    }

    // Right click: trajectory inspector.
    if buttons.just_pressed(MouseButton::Right) {
        if let Some(texel) = texel {
            inspect.write(InspectRequest(Some(settings.texel_to_world(texel))));
        }
    }
}

pub fn keyboard_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<Settings>,
    mut auto: ResMut<AutoZoom>,
    mut verify: ResMut<Verify>,
    mut inspect: MessageWriter<InspectRequest>,
    mut help: ResMut<crate::ui::ShowHelp>,
    mut commands: Commands,
    sim: Res<SimState>,
    stats: Res<GpuStats>,
) {
    let modes = [
        (KeyCode::Digit1, ColorMode::Fate),
        (KeyCode::Digit2, ColorMode::EjectionTime),
        (KeyCode::Digit3, ColorMode::EnergyError),
        (KeyCode::Digit4, ColorMode::Steps),
        (KeyCode::Digit5, ColorMode::Sensitivity),
    ];
    for (key, mode) in modes {
        if keys.just_pressed(key) {
            settings.color_mode = mode;
            if mode == ColorMode::Sensitivity {
                settings.sensitivity = true;
            }
        }
    }
    if keys.just_pressed(KeyCode::KeyS) {
        settings.sensitivity = !settings.sensitivity;
        if !settings.sensitivity && settings.color_mode == ColorMode::Sensitivity {
            settings.color_mode = ColorMode::Fate;
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        let keep = (settings.color_mode, settings.mass_preset);
        *settings = Settings::default();
        (settings.color_mode, settings.mass_preset) = keep;
        auto.active = false;
    }
    if keys.just_pressed(KeyCode::KeyM) {
        settings.mass_preset = (settings.mass_preset + 1) % MASS_PRESETS.len();
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        settings.physics.eta = (settings.physics.eta / 1.5).max(1e-4);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        settings.physics.eta = (settings.physics.eta * 1.5).min(0.2);
    }
    if keys.just_pressed(KeyCode::Minus) {
        settings.physics.t_max_cr = (settings.physics.t_max_cr / 1.5).max(5.0);
    }
    if keys.just_pressed(KeyCode::Equal) {
        settings.physics.t_max_cr = (settings.physics.t_max_cr * 1.5).min(2000.0);
    }
    if keys.just_pressed(KeyCode::Space) {
        zoom_into_boundary(&mut settings, &stats, &sim);
        auto.active = false;
    }
    if keys.just_pressed(KeyCode::Backspace) {
        settings.half_width = (settings.half_width * ZOOM_STEP).min(DEFAULT_HALF_WIDTH * 4.0);
        auto.active = false;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        if auto.active {
            auto.active = false;
        } else {
            let keep = (settings.color_mode, settings.mass_preset, settings.physics);
            *settings = Settings::default();
            (settings.color_mode, settings.mass_preset, settings.physics) = keep;
            *auto = AutoZoom {
                active: true,
                steps_left: TOUR_STEPS,
                wait_until: f64::INFINITY,
            };
        }
    }
    if keys.just_pressed(KeyCode::KeyV) {
        verify.pending = true;
    }
    if keys.just_pressed(KeyCode::Escape) {
        inspect.write(InspectRequest(None));
    }
    if keys.just_pressed(KeyCode::KeyH) {
        help.0 = !help.0;
    }
    if keys.just_pressed(KeyCode::F12) || keys.just_pressed(KeyCode::KeyP) {
        let path = format!("threebody-{:.0}x-{}.png", settings.zoom_factor(), sim.generation);
        info!("saving screenshot {path}");
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
    }
}

/// Picks a 64x64 stats cell on a basin boundary (fates mixed, with an almost single-colour
/// cell within two cells of it, mildly preferring the middle of the view) and zooms 5x
/// onto it.
pub fn zoom_into_boundary(settings: &mut Settings, stats: &GpuStats, sim: &SimState) -> bool {
    if !stats.valid || stats.generation != sim.generation || stats.grid.len() != GRID * GRID {
        return false;
    }
    let cell = |x: i32, y: i32| stats.grid[y as usize * GRID + x as usize];
    let entropy = |c: [u32; 4]| -> f64 {
        let ejected: u32 = c[..3].iter().sum();
        if ejected == 0 || ejected * 2 < ejected + c[3] {
            return -1.0;
        }
        c[..3]
            .iter()
            .filter(|&&k| k > 0)
            .map(|&k| {
                let p = k as f64 / ejected as f64;
                -p * p.ln()
            })
            .sum()
    };
    let pure = |c: [u32; 4]| {
        let n: u32 = c.iter().sum();
        n > 0 && c[..3].iter().any(|&k| k as f64 >= 0.9 * n as f64)
    };
    let g = GRID as i32;
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for cy in 2..g - 2 {
        for cx in 2..g - 2 {
            let e = entropy(cell(cx, cy));
            if e < 0.3 {
                continue;
            }
            let near_basin = (-2..=2)
                .flat_map(|dy| (-2..=2).map(move |dx| (dx, dy)))
                .any(|(dx, dy)| pure(cell(cx + dx, cy + dy)));
            let d = Vec2::new(cx as f32 + 0.5, cy as f32 + 0.5) / GRID as f32 - 0.5;
            let score = e + if near_basin { 1.0 } else { 0.0 } - 0.6 * d.length() as f64;
            if score > best_score {
                best_score = score;
                best = Some(Vec2::new(cx as f32 + 0.5, cy as f32 + 0.5));
            }
        }
    }
    let Some(cell) = best else { return false };
    let texel = cell / GRID as f32 * TEX_SIZE as f32;
    settings.center = settings.texel_to_world(texel);
    settings.half_width /= ZOOM_STEP;
    true
}

pub fn run_auto_zoom(
    mut auto: ResMut<AutoZoom>,
    mut settings: ResMut<Settings>,
    sim: Res<SimState>,
    stats: Res<GpuStats>,
    time: Res<Time<Real>>,
) {
    if !auto.active || !sim.done {
        return;
    }
    let now = time.elapsed_secs_f64();
    if auto.steps_left == 0 {
        auto.active = false;
        return;
    }
    if now < auto.wait_until {
        return;
    }
    if zoom_into_boundary(&mut settings, &stats, &sim) {
        auto.steps_left -= 1;
        auto.wait_until = f64::INFINITY;
    }
}

/// Restarts the pause timer once a tour step's image is complete.
pub fn auto_zoom_pause(mut auto: ResMut<AutoZoom>, sim: Res<SimState>, time: Res<Time<Real>>) {
    if auto.active && sim.done && auto.wait_until.is_infinite() {
        auto.wait_until = time.elapsed_secs_f64() + 1.5;
    }
}
