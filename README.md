# Three-body ejection fractal (Bevy 0.20 + WGSL compute)

Every pixel is its own three-body problem. Three equal stars start at rest: stars 0 and 1
always start at (−1, 0) and (1, 0), and star 2 starts at the pixel. The pixel is
coloured by which star is ejected (gold / blue / red), or dark if nobody has left after
200 crossing times. This is the map from Popperish's
[*We Threw Three Stars Together 5 Million Times*](https://www.youtube.com/watch?v=gvSyJ3kQSHU)
(transcript in `transcript.md`, original plan in `PLAN.md`).

![zoom tour 1x to 625x](docs/zoom-tour.jpg)

```bash
cargo run --release
```

The first 1024² image takes about 12 s on an M5. It refines progressively at 256² → 512² → 1024².

## Controls

| Key | |
|---|---|
| wheel / drag | zoom about cursor / pan (old image is warped in as a preview) |
| right-click / Esc | replay that pixel's orbit on the CPU (f64) / close |
| Space / Backspace | zoom 5× into a basin boundary / zoom out 5× |
| T | auto tour: 5× four times, to 625× |
| 1–5 | colour: fate, ejection time, energy error, step count, sensitivity |
| S | rerun every pixel at η/2 alongside (needed for mode 5) |
| `[` `]` / `-` `=` | step size η / cutoff t_max |
| M | masses: 1-1-1, 3-4-5, 1-2-3 |
| V | check 400 random GPU pixels against the f64 CPU reference |
| P | screenshot · H hide help · R reset |

Unattended runs: `--mode N --masses N --sensitivity --tour --shot PREFIX --verify --exit`.

## Physics

From rest, total angular momentum is zero, so near-collisions (r ~ 1e-5) are common.
Adaptive RK4 can't integrate through them, and f32 absolute coordinates can't represent them. So both
the GPU kernel and the f64 reference (`src/physics_cpu.rs`) use:

- **Jacobi coordinates about the closest pair**, switched when another pair gets closer.
  Close separations are stored directly instead of as differences of large positions.
- **The logarithmic-Hamiltonian leapfrog** (Mikkola & Tanikawa 1999), which regularises
  two-body collisions, composed to 4th order (Yoshida). Its step shrinks as √(r_min/r_v), so tight
  binaries keep a fixed number of steps per orbit.
- **The escape test:** a star is ejected when it has positive energy relative to the other two,
  is receding, is beyond 5 virial radii, and leaves a bound pair behind.

Checks (`cargo test --release --lib`, `cargo run --release --example tune`):

- The Pythagorean 3-4-5 problem ejects the lightest star.
- The default view splits 32.4 / 32.1 / 33.4%, with 2% undecided (video: ~32% each, 4%).
- Halving η leaves fates unchanged for 98.7% of early (<10 t_cr) ejections and 40% of
  later ones (video: 98–99.9% and 38–46%).
- GPU f32 vs CPU f64 agreement: 100% for early ejections and about 37% for later ones.

## Limits

The GPU works in f32, so neighbouring pixels stop having distinct starting positions at
roughly 10⁴× zoom. The zoom is clamped there and the overlay says so. Going deeper would need
double-float arithmetic in the dynamics, not just in the pixel coordinates.
