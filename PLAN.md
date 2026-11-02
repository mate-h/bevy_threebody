# Plan: Three-Body Ejection Fractal — Bevy 0.20 + WGSL compute

Goal: reproduce the "fate map" from Popperish's *We Threw Three Stars Together 5 Million
Times* (https://www.youtube.com/watch?v=gvSyJ3kQSHU) as a real-time, zoomable fractal.

## 1. What the video actually computes

- Three **equal** point masses, Newtonian gravity only, no collisions, no softening.
- All start **at rest**. Stars A and B always start at the same two fixed points.
- Star C starts **at the pixel's position** in the plane.
- Integrate until one star is **ejected**; colour the pixel by which one:
  gold / blue / red. If nobody has left by the cutoff → "undecided" (~4% of pixels).
- First image is 1024×1024 (1,048,576 simulations). Expected split: ~32% each, ~4% undecided.
- Big smooth basins separated by boundaries where the colours interleave; zooming 5×, 25×,
  …, 625× keeps revealing more boundary (fractal basin boundaries).
- Fast ejections (first few swings) are robust to integrator accuracy (98–99.9% agree).
  Long-lived systems (~4× longer) are essentially unpredictable (38–46% self-agreement).

Out of scope for v1 (other parts of the video): unequal-mass statistics, planet survival
in hierarchical triples, 3D figure-8 stability, Alpha/Proxima Centauri. Some are cheap
add-ons later (§9).

## 2. Physics model

Units: G = 1, m = 1 each (configurable masses later). Default fixed stars at
A = (−1, 0), B = (1, 0); view initially covers roughly x ∈ [−3, 3], y ∈ [−3, 3].
Bilateral symmetry (y → −y) is a free sanity check on the image.

**State per pixel**: positions and velocities of 3 stars (12 f32), time t, step size dt,
status/result, step count.

**Integrator (GPU, f32)**: close encounters are everything here, so the step size must be
adaptive per pixel.
- v1: **RK4 with a dynamical-time step**
  `dt = η · min_ij sqrt(r_ij³ / (m_i + m_j))`, η ≈ 0.01 (uniform-ish error per orbit),
  clamped to `[dt_min, dt_max]`.
- v2 option: embedded RK45 (Dormand–Prince) with error control, or 4th-order Hermite.
  Keep the integrator behind one WGSL function so it can be swapped and A/B tested.
- Track energy E; store |ΔE/E| at end for a diagnostics overlay.
- Pixels where C starts on top of A or B: the step size clamp keeps them finite;
  they'll be classified like anything else (or marked "singular" if dt hits dt_min).

**Escape test** (every K steps, for each star k with partner pair i,j):
- pair COM `R, V`, `M = m_i + m_j`; relative `r = x_k − R`, `v = v_k − V`
- `E_k = ½ μ |v|² − G m_k M / |r|`, μ = m_k M / (m_k + M)
- ejected if `E_k > 0` **and** `r·v > 0` **and** `|r| > R_far` (e.g. 10× initial size)
  **and** the remaining pair is bound.
- Result = k, plus ejection time t_ej (used for shading).

**Cutoff**: `T_max` in crossing times. For a system starting at rest,
`t_cr = G·M_tot^{5/2} / (2|E|)^{3/2}`. Default T_max ≈ 100 t_cr, configurable; tune until
undecided ≈ 4% on the default view.

**Precision caveat (macOS)**: Metal has no f64 in shaders, so the GPU kernel is f32.
That's fine for the picture (the video itself says long-lived pixels are chaotic noise
anyway), but:
- the view centre / pixel coordinates need care beyond ~1e5× zoom — plan a
  **double-float (df64) pixel-coordinate** path for deep zoom (only the initial position
  needs it; dynamics stays f32);
- the CPU f64 reference integrator (§6) is the ground truth.

## 3. GPU architecture

### Persistent, time-sliced simulation
Don't integrate a whole pixel in one dispatch — long runs would trip the macOS GPU
watchdog and freeze the UI. Instead:

- **Storage buffer** `states: array<PixelState>` (≈64 B/pixel → 64 MB at 1024²).
- Each frame, dispatch `step` over all pixels; each thread advances its pixel by up to
  `steps_per_frame` (adaptive, tuned from frame time) or until it finishes.
- A `generation` counter in the params uniform: when the view changes, `init` re-seeds
  every pixel; stale threads can't write old results.
- Optional: an atomic "pixels still running" counter → progress bar, and stop
  dispatching when it hits 0.

### Kernels (one WGSL file, multiple entry points)
1. `init` — compute pixel world position from view params, set initial state.
2. `step` — integrate + escape test, write result when done.
3. `shade` — write `rgba8` storage texture from states (colour by result, brightness by
   log t_ej, undecided = near-black, running pixels = dim preview of current leader or
   previous frame's colour).

Workgroup size 8×8. Divergence between neighbouring pixels is expected and acceptable.

### Progressive refinement for zooming
- Render first at 1/4 resolution (256²), then 1/2, then full; each level is its own
  `init` pass at that resolution (or a coarse-to-fine pixel ordering).
- While a new view computes, show the **previous texture reprojected** (scaled/panned in
  the display shader) so the zoom feels instant.

### Bevy 0.20 integration
Follow the structure of Bevy's `examples/shader/compute_shader_game_of_life.rs` **at the
v0.20 tag** — the render-graph / render-schedule APIs change between releases, so copy the
idioms from that example rather than from memory.

- **Main world**
  - `FractalParams` resource (view centre, scale, resolution, masses, A/B positions,
    η, T_max, colour mode, generation). Derive `ExtractResource`, `ShaderType`.
  - Output `Image` created with `TextureUsages::STORAGE_BINDING | TEXTURE_BINDING |
    COPY_DST`, shown fullscreen via a `Sprite` (2D camera) or a UI `ImageNode`.
  - Input systems: mouse-wheel zoom about cursor, drag to pan, keys for presets,
    reset, T_max/η, palette, "auto-zoom".
- **Render world** (`RenderApp` sub-app)
  - `ExtractResourcePlugin::<FractalParams>` and `ExtractResourcePlugin::<FractalImage>`.
  - Pipeline resource: bind group layout (uniform params, storage buffer states,
    storage texture output), `queue_compute_pipeline` for `init`, `step`, `shade`.
  - Prepare system: (re)allocate the states buffer on resolution change, write the
    uniform, build the bind group.
  - Render node/system: if `generation` changed → dispatch `init`; then `step`; then
    `shade`. Order before the camera's main pass.
- Readback (for stats / auto-zoom): use Bevy's `Readback` component on a small
  result-histogram buffer (or the image) — async, a frame or two late is fine.

## 4. Visuals

- Palette: gold (A), blue (B), red (C) as in the video; undecided dark grey.
- Brightness = f(log t_ej): fast ejections bright, late ones darker → boundaries pop.
- Modes (key toggle): fate, ejection time heatmap, energy-error heatmap, **sensitivity**
  (see §5), step-count heatmap.
- Overlay (bevy_ui text): zoom factor, centre, % per colour, % undecided,
  progress, steps/s.
- Mark A and B starting positions with small dots.

## 5. Video-inspired extras (cheap once the core works)

- **Sensitivity map**: run each pixel twice (η and η/10, or η·10) in parallel lanes;
  colour where they disagree. Should reproduce "fast = robust, slow = coin flip".
- **Auto-zoom tour**: pick the highest-entropy 3×3 neighbourhood from a downsampled
  readback, zoom 5× toward it, repeat (5 steps → 625×, like the video).
- **Stats panel**: histogram of ejection times in crossing-times (video: half of
  equal-mass systems eject within ~17 t_cr).
- **Trajectory inspector**: click a pixel → run that system on the CPU in f64 and draw
  the three orbits with gizmos, animated.

## 6. Correctness / tests

- `src/physics_cpu.rs`: f64 reference integrator with the same escape test.
- `cargo test`:
  - **Pythagorean problem** (Burrau; Szebehely & Peters 1967): masses 3, 4, 5 at rest on
    a 3-4-5 triangle → mass-3 ejected, 4+5 leave as a bound binary.
  - Two-body circular orbit energy/period conservation.
  - Symmetry: pixel (x, y) and (x, −y) give the same result.
- GPU vs CPU: sample ~1000 random pixels, compare fates; expect near-total agreement for
  early ejections, chance-level for late ones (that's the point).
- Default-view stats ≈ 32/32/32/4 — if far off, revisit T_max / escape test.

## 7. Project layout

```
Cargo.toml                 bevy = "0.20"
assets/shaders/threebody.wgsl   init / step / shade entry points, integrator, escape test
src/main.rs                app setup, window, camera, fullscreen sprite
src/params.rs              FractalParams, presets
src/input.rs               zoom / pan / keys / auto-zoom
src/render/mod.rs          FractalComputePlugin (RenderApp wiring)
src/render/pipeline.rs     layouts, pipelines, bind groups, buffers
src/render/dispatch.rs     node/system that dispatches init/step/shade
src/ui.rs                  overlay text, stats
src/physics_cpu.rs         f64 reference + inspector + tests
```

## 8. Milestones

1. **Skeleton** — Bevy 0.20 app with fullscreen storage texture written by a trivial
   compute shader (gradient). Confirms the render-world plumbing.
2. **CPU reference** — f64 integrator + escape test + Pythagorean test passing.
3. **GPU fate map** — `init`/`step`/`shade` with persistent state at 512², single view.
   Compare against CPU on sampled pixels.
4. **Interaction** — zoom/pan, generation reset, progressive resolution, reprojected
   preview, 1024².
5. **Polish** — palette/brightness by t_ej, overlay stats, presets for the video's
   5-step zoom sequence.
6. **Extras** — sensitivity map, auto-zoom, trajectory inspector, df64 coordinates for
   deep zoom.

## 9. Later

- Unequal masses (3-4-5 → ~61% lightest ejected; 1-2-3 → ~84%; compare to the
  ∝ 1/m³ rule of thumb) — only needs a masses uniform + stats.
- Monte-Carlo mode: random initial conditions instead of a grid, for the probability
  experiments.

## Open questions

- Exact starting positions of A and B in the video aren't stated; we pick (±1, 0) and
  expose them as parameters.
- T_max isn't stated; tune to the ~4% undecided figure.

## As built (deviations from this plan)

- **Integrator:** adaptive RK4 failed. From rest the system has zero angular momentum, so
  near-collisions to r ~ 1e-5 are common, and more than 10% of runs blew up energy even in f64.
  It was replaced on both CPU and GPU by a logH leapfrog (Mikkola–Tanikawa) with Yoshida
  4th-order composition, in Jacobi coordinates about the closest pair, with steps scaled by
  √(r_min/r_v). See README.
- **Cutoff and escape radius:** tuned to t_max = 200 t_cr and r_far = 5 r_v. That gives
  32/32/33% with 2% undecided.
- **df64 coordinates:** dropped. Higher-precision starting positions don't help while the
  dynamics stay in f32, because the difference is rounded away at the first step. The app
  instead clamps zoom at the f32 limit (about 10⁴×) and says so. Real deep zoom needs df64
  dynamics.
- **Refinement levels:** each level (256/512/1024) is reseeded from scratch rather than
  reusing coarse pixels (about 30% extra work).
