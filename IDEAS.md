# Possibilities: Barnes–Hut, improvements, and 3D

Inputs: the current app (three-body fate map, see README) and Deadlock's
[*How to make HUGE N-Body Simulations*](https://www.youtube.com/watch?v=nZHjD3cI-EU).

## 1. How the two problems relate

| | Current app | Barnes–Hut video |
|---|---|---|
| Shape | ~1M **independent** 3-body systems | **one** system of ~1M bodies |
| Parallelism | one GPU thread per system, no communication | every body depends on every other |
| Hard part | close encounters, precision, chaos | O(N²) force sum |
| Accuracy | regularised, energy error ~1e-5 | approximate forces (θ), softening ε |
| Output | an image of outcomes | an animation of one evolving system |

So Barnes–Hut can't make the fractal itself faster. A tree only pays off when N is large,
and here N = 3 per pixel. What carries over is the **infrastructure**:

- **Bevy compute plumbing:** pipelines, bind groups, ping-pong textures, readback,
  time-sliced dispatch with an adaptive step budget.
- **Interaction and UI:** zoom/pan, stats overlay, screenshots, CLI autopilot.
- **Verification pattern:** an f64 CPU reference, plus GPU-vs-CPU comparison on random samples.
- **Integrator know-how:** the logH leapfrog is a leapfrog. Barnes–Hut sims normally use
  kick-drift-kick leapfrog, so the code style and the tests transfer.

The video stops at "run it on the GPU is left as an exercise". That step is exactly what
this codebase is set up for.

## 2. Option A: add a GPU Barnes–Hut "galaxy" mode (the video's sim, finished)

A second simulation mode in the same app, sharing the render plugin structure.

**GPU tree build.** The video's insert-one-at-a-time build is serial. On the GPU the
standard approach is:

1. Compute a bounding box with a reduction pass.
2. Give each body a Morton code (Z-order: interleaved x/y bits). This is also the
   video's "sort for cache locality" fix, done properly.
3. Radix sort by Morton code. That needs a few passes; WGSL has no built-in sort.
4. Build the tree in parallel from the sorted codes (Karras 2012 "linear BVH", or a
   linear quadtree from shared code prefixes).
5. Compute centre of mass bottom-up. WGSL has no float atomics, so use per-node
   "children done" `atomic<u32>` counters: the last child to finish computes its parent
   and walks up.

**Force pass.** Use the video's stackless traversal with "next" pointers. This is ideal for
GPU threads, because there is no per-thread stack. Bodies are sorted by Morton code, so
neighbouring threads walk similar paths through the tree (coherent memory access, less
divergence).

**Integrator.** Kick-drift-kick leapfrog with softening ε.

**Rendering.** There are two routes:
- **Instanced point sprites** with additive blending and HDR bloom.
- **A splat pass** into a storage texture. This needs fixed-point `atomicAdd`, again
  because WGSL has no float atomics.

**Stepping stones.** A tiled direct sum (O(N²) with workgroup shared memory) is a good first
version and a correctness reference. Expect it to handle around 10⁴–10⁵ bodies
interactively. Then use Barnes–Hut for 10⁶.

**Collisions (from the video).** Use a spatial hash grid on the GPU. Merge colliding bodies,
or solve them as soft contacts.

**Effort:** large. Tree build and sort are the bulk of it.

## 3. Option B: improve the current fractal sim

Roughly in order of value per effort:

1. **Compact the running pixels (S–M).**
   - *Problem:* at the 1024² level most of the ~9 s is the long tail. A few percent of
     pixels are still running while every thread in their workgroups sits idle.
   - *Fix:* each frame, compact the indices of running pixels into a list (prefix sum or
     atomic append), then dispatch only over that list.
   - *Payoff:* likely a 2–4× faster finish.
2. **Reuse coarse levels (S).**
   - Line up the sample positions so each coarse pixel coincides with one fine pixel, then
     copy those results instead of recomputing them.
   - Saves the ~30% overhead of reseeding each level.
3. **Adaptive refinement (M).**
   - At deeper zooms most of the frame is smooth basin. Only compute fine pixels whose
     coarse neighbours disagree, and fill the rest.
   - *Risk:* missing small islands. Guard it by always computing pixels near disagreement
     and pixels that ejected late.
4. **A collision outcome (S).**
   - Give stars a radius and add a fourth "collided" colour. The video notes that real
     stars "would crash".
   - Much of the chaotic sea would turn into collisions. That's interesting physics, and it
     also takes the hardest runs off the GPU's hands.
5. **Deep zoom with double-float dynamics (M–L).**
   - Run the integrator in df64 (pairs of f32), roughly 10–20× slower, only when the zoom
     passes the f32 limit.
   - Needs a check that Metal doesn't fuse or reorder the error-free float tricks that
     df64 relies on.
6. **Use the logH engine for the first video's other experiments (S–M each):**
   - *Planet safe zones:* thousands of massless planets in a hierarchical triple. Each
     planet needs only 3 force terms, no tree. A planet's orbit radius and phase make a
     natural 2D survival map, like the Holman–Wiegert boundaries in the video.
   - *Unequal masses:* Monte Carlo ejection probabilities, to compare against the 1/m³
     rule of thumb.
   - *Proxima Centauri:* the 10-Myr run. It stays hierarchical, which is cheap.

## 4. Option C: bring them to 3D

### 4a. The three-body fate map in 3D: a catch

If all three stars **start at rest**, they stay in the plane through their three starting
points forever. A 3D start position for star C is therefore the 2D problem rotated about the
A–B axis:

`fate3D(x, y, z) = fate2D(x, √(y² + z²))`

That is a solid of revolution with no new information. Pretty to ray-march, but cheap to fake.

Genuinely 3D maps need out-of-plane motion. Some ways to get it:

- **Kick maps:** pixel = (C's position in the plane, C's out-of-plane speed v_z), giving a
  3D volume. Or fix the position and map two velocity components. This is the first video's
  "kick the figure-8 out of the plane" test turned into a picture.
- **Rotating starts:** give the A–B pair some spin about an axis that isn't perpendicular
  to the plane.

**Code changes:**
- The logH and Jacobi integrator generalises directly: `vec2` → `vec3`.
- Per-pixel state grows from 48 to 64 bytes.
- The escape test and pair switching are unchanged.

**Rendering, two options:**
- **A movable slice plane.** Each slice is a 2D image exactly like today. Cheap and
  incremental.
- **A 3D fate volume.** For example 256³ ≈ 16.7M systems, about 16× one 1024² image, so
  a few minutes on the M5. Render it with front-to-back ray-marching, and either make the
  chaotic sea transparent or show only the boundaries.

**Inspector:** draw the 3D orbits with gizmos under a `Camera3d` with orbit controls.

### 4b. Barnes–Hut in 3D

- **Tree:** quadtree → octree (8 children). The "children stored contiguously" layout and
  the "next" pointer traversal still work, and so do 3D Morton codes.
- **Rendering:** a 3D orbit camera, additive point sprites, and bloom suit galaxies.
- **Cost:** about the same as 2D per body, with somewhat deeper trees.

### 4c. Combined

A star cluster in 3D under Barnes–Hut, where close few-body encounters are handed off to
the regularised three-body integrator. Real codes do this (NBODY6 regularises binaries
separately). It's the natural endpoint of both videos: Barnes–Hut for the crowd, and the
current engine for the moments that decide who gets ejected. **Effort:** very large.

## 5. Suggested path

1. **B1 (running-pixel compaction) and B2 (coarse reuse).** Quick wins on the existing app.
2. **C 4a kick map with a slice plane.** This is the first genuinely 3D result, and it reuses
   nearly everything: `vec3` integrator, one extra parameter axis, orbit-camera inspector.
3. **A, with a direct-sum stepping stone and then Barnes–Hut.** This is new subsystems
   (sort, tree, point rendering). Built as a second mode in the same app, it reuses the
   compute plugin pattern and UI.
4. **3D octree and bloom, then the hybrid (4c)** if you want to keep going.

Shared refactor before step 3: split `render/` into a generic "compute sim" layer plus
per-mode pipelines, and move the WGSL physics into importable shader modules (Bevy
supports `#import`).
