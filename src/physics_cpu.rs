//! f64 reference implementation of the three-body ejection experiment.
//!
//! This mirrors `assets/shaders/threebody.wgsl` step for step (same coordinates, integrator,
//! escape test and cutoff) so GPU results can be checked against it. Units: G = 1.
//!
//! Starting from rest means zero angular momentum, so near-collisions (r ~ 1e-5) are common.
//! Plain adaptive RK4 can't get through them, and in f32 absolute coordinates can't even
//! represent them. So:
//! - the state is held in Jacobi coordinates relative to the closest pair (`q` = separation
//!   of the pair, `qq` = third star relative to the pair's centre of mass), re-chosen when
//!   another pair becomes closer. Close separations are stored directly, not as a difference
//!   of two large positions;
//! - the integrator is the logarithmic-Hamiltonian leapfrog of Mikkola & Tanikawa (1999) /
//!   Preto & Tremaine (1999), which follows two-body collisions exactly, composed to 4th
//!   order with Yoshida's triple jump.

pub type V2 = [f64; 2];

/// Parameters shared with the GPU kernel (see `GpuParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub masses: [f64; 3],
    /// Fixed starting positions of stars 0 and 1. Star 2 starts at the pixel.
    pub pos_a: V2,
    pub pos_b: V2,
    /// Step size in regularised time s, as a fraction of |E0| * t_cr: roughly 1/eta steps
    /// per crossing time while nothing is close.
    pub eta: f64,
    /// Cutoff in crossing times.
    pub t_max_cr: f64,
    /// Escape distance in units of the virial radius.
    pub r_far: f64,
    /// Hard cap on integration steps; a run that hits it counts as undecided.
    pub max_steps: u32,
    /// Shrink the step as sqrt(r_min / r_v) so tight binaries keep a fixed number of
    /// steps per orbit.
    pub resolve_binaries: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            masses: [1.0, 1.0, 1.0],
            pos_a: [-1.0, 0.0],
            pos_b: [1.0, 0.0],
            eta: 0.02,
            t_max_cr: 200.0,
            r_far: 5.0,
            max_steps: 200_000,
            resolve_binaries: true,
        }
    }
}

/// How often (in steps) the escape test runs. Must match `ESCAPE_CHECK_EVERY` in the shader.
pub const ESCAPE_CHECK_EVERY: u32 = 4;
/// Switch the Jacobi pair when another pair is closer than this fraction of the current
/// one. Must match the shader.
pub const SWITCH_RATIO: f64 = 0.8;

#[derive(Clone, Copy, Debug)]
pub struct State {
    /// The third star; the close pair is the other two, `pair(k)`.
    pub k: usize,
    /// Pair separation x_j - x_i and its rate.
    pub q: V2,
    pub u: V2,
    /// Third star relative to the pair's centre of mass, and its rate.
    pub qq: V2,
    pub uu: V2,
    pub t: f64,
    /// Centre of mass (constant, since total momentum is zero). Only used for output.
    pub com: V2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fate {
    /// Index of the ejected star.
    Ejected(usize),
    /// Nobody left before the cutoff (or the step cap was hit).
    Undecided,
}

#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub fate: Fate,
    /// Time of ejection (or of the cutoff), in crossing times.
    pub t_cr_units: f64,
    pub steps: u32,
    /// |E - E0| / |E0| at the end of the run.
    pub energy_err: f64,
}

fn add(a: V2, b: V2) -> V2 {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn mul(a: V2, s: f64) -> V2 {
    [a[0] * s, a[1] * s]
}

fn dot(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn len(a: V2) -> f64 {
    dot(a, a).sqrt()
}

/// The pair (i, j) left when star k is the third.
pub fn pair(k: usize) -> (usize, usize) {
    ((k + 1) % 3, (k + 2) % 3)
}

/// Positions and velocities in the centre-of-mass frame, indexed by star.
pub fn com_frame(s: &State, m: &[f64; 3]) -> ([V2; 3], [V2; 3]) {
    let (i, j) = pair(s.k);
    let mp = m[i] + m[j];
    let mt = mp + m[s.k];
    let mut x = [[0.0; 2]; 3];
    let mut v = [[0.0; 2]; 3];
    let cp = mul(s.qq, -m[s.k] / mt);
    let cv = mul(s.uu, -m[s.k] / mt);
    x[i] = sub(cp, mul(s.q, m[j] / mp));
    x[j] = add(cp, mul(s.q, m[i] / mp));
    x[s.k] = mul(s.qq, mp / mt);
    v[i] = sub(cv, mul(s.u, m[j] / mp));
    v[j] = add(cv, mul(s.u, m[i] / mp));
    v[s.k] = mul(s.uu, mp / mt);
    (x, v)
}

pub fn absolute_positions(s: &State, m: &[f64; 3]) -> [V2; 3] {
    let (x, _) = com_frame(s, m);
    x.map(|p| add(p, s.com))
}

/// Separations from the pair to the third star: (x_k - x_i, x_k - x_j).
fn third_separations(s: &State, m: &[f64; 3]) -> (V2, V2) {
    let (i, j) = pair(s.k);
    let mp = m[i] + m[j];
    (add(s.qq, mul(s.q, m[j] / mp)), sub(s.qq, mul(s.q, m[i] / mp)))
}

/// U = sum m_i m_j / r_ij (positive).
fn potential(s: &State, m: &[f64; 3]) -> f64 {
    let (i, j) = pair(s.k);
    let (rik, rjk) = third_separations(s, m);
    m[i] * m[j] / len(s.q) + m[i] * m[s.k] / len(rik) + m[j] * m[s.k] / len(rjk)
}

fn kinetic(s: &State, m: &[f64; 3]) -> f64 {
    let (i, j) = pair(s.k);
    let mp = m[i] + m[j];
    let mt = mp + m[s.k];
    0.5 * (m[i] * m[j] / mp) * dot(s.u, s.u) + 0.5 * (m[s.k] * mp / mt) * dot(s.uu, s.uu)
}

pub fn energy(s: &State, m: &[f64; 3]) -> f64 {
    kinetic(s, m) - potential(s, m)
}

/// Second time derivatives of (q, qq).
fn accel(s: &State, m: &[f64; 3]) -> (V2, V2) {
    let (i, j) = pair(s.k);
    let mp = m[i] + m[j];
    let mt = mp + m[s.k];
    let (rik, rjk) = third_separations(s, m);
    let f = |r: V2| mul(r, 1.0 / (dot(r, r) * len(r)));
    let (fq, fik, fjk) = (f(s.q), f(rik), f(rjk));
    let aq = add(mul(fq, -mp), mul(sub(fjk, fik), m[s.k]));
    let aqq = mul(add(mul(fik, m[i]), mul(fjk, m[j])), -mt / mp);
    (aq, aqq)
}

/// One logH leapfrog step of length h in regularised time; `b` = -E0.
fn leapfrog(s: &mut State, m: &[f64; 3], h: f64, b: f64) {
    let drift = |s: &mut State, h: f64| {
        let dt = h / (kinetic(s, m) + b);
        s.q = add(s.q, mul(s.u, dt));
        s.qq = add(s.qq, mul(s.uu, dt));
        s.t += dt;
    };
    drift(s, 0.5 * h);
    let dt = h / potential(s, m);
    let (aq, aqq) = accel(s, m);
    s.u = add(s.u, mul(aq, dt));
    s.uu = add(s.uu, mul(aqq, dt));
    drift(s, 0.5 * h);
}

/// Yoshida (1990) 4th-order triple jump weights.
const YOSHIDA_W1: f64 = 1.351_207_191_959_657_6;
const YOSHIDA_W0: f64 = -1.702_414_383_919_315_3;

fn step(s: &mut State, m: &[f64; 3], h: f64, b: f64) {
    leapfrog(s, m, YOSHIDA_W1 * h, b);
    leapfrog(s, m, YOSHIDA_W0 * h, b);
    leapfrog(s, m, YOSHIDA_W1 * h, b);
}

/// All three separations and their rates, indexed by the star *opposite* each pair:
/// `r[c]` = x_b - x_a with (a, b) = pair(c). Built from `q` and `qq` only, never from
/// differences of large positions.
fn separations(s: &State, m: &[f64; 3]) -> ([V2; 3], [V2; 3]) {
    let (i, j) = pair(s.k);
    let mp = m[i] + m[j];
    let (rik, rjk) = third_separations(s, m);
    let wik = add(s.uu, mul(s.u, m[j] / mp));
    let wjk = sub(s.uu, mul(s.u, m[i] / mp));
    let mut r = [[0.0; 2]; 3];
    let mut w = [[0.0; 2]; 3];
    // pair(k) = (i, j); pair(i) = (j, k); pair(j) = (k, i).
    (r[s.k], w[s.k]) = (s.q, s.u);
    (r[i], w[i]) = (rjk, wjk);
    (r[j], w[j]) = (mul(rik, -1.0), mul(wik, -1.0));
    (r, w)
}

/// Re-expresses the state with star `c` as the third.
fn with_third(s: &State, m: &[f64; 3], c: usize) -> State {
    let (r, w) = separations(s, m);
    let (a, b) = pair(c);
    let mab = m[a] + m[b];
    // x_c - com_ab = (m_b (x_c - x_b) - m_a (x_a - x_c)) / m_ab = (m_b r[a] - m_a r[b]) / m_ab
    let qq = mul(sub(mul(r[a], m[b]), mul(r[b], m[a])), 1.0 / mab);
    let uu = mul(sub(mul(w[a], m[b]), mul(w[b], m[a])), 1.0 / mab);
    State { k: c, q: r[c], u: w[c], qq, uu, t: s.t, com: s.com }
}

fn min_separation(s: &State, m: &[f64; 3]) -> f64 {
    let (rik, rjk) = third_separations(s, m);
    len(s.q).min(len(rik)).min(len(rjk))
}

/// Keeps the closest pair as the Jacobi pair.
fn maybe_switch(s: &mut State, m: &[f64; 3]) {
    let (i, j) = pair(s.k);
    let (rik, rjk) = third_separations(s, m);
    let (dq, dik, djk) = (len(s.q), len(rik), len(rjk));
    if dik < SWITCH_RATIO * dq && dik <= djk {
        // New pair (i, k): third is j.
        *s = with_third(s, m, j);
    } else if djk < SWITCH_RATIO * dq {
        *s = with_third(s, m, i);
    }
}

/// Crossing time and virial radius of a system with total energy `e0` (< 0).
pub fn scales(m: &[f64; 3], e0: f64) -> (f64, f64) {
    let mt: f64 = m.iter().sum();
    let t_cr = mt.powf(2.5) / (2.0 * e0.abs()).powf(1.5);
    let pair_sum = m[0] * m[1] + m[0] * m[2] + m[1] * m[2];
    (t_cr, pair_sum / e0.abs())
}

/// Returns the star that has escaped: positive energy relative to the other two, moving
/// away from them, beyond `r_far_abs`, and leaving a bound pair behind.
pub fn escaped(s: &State, m: &[f64; 3], r_far_abs: f64) -> Option<usize> {
    let (x, v) = com_frame(s, m);
    for k in 0..3 {
        let (i, j) = pair(k);
        let mp = m[i] + m[j];
        let com = mul(add(mul(x[i], m[i]), mul(x[j], m[j])), 1.0 / mp);
        let vcom = mul(add(mul(v[i], m[i]), mul(v[j], m[j])), 1.0 / mp);
        let r = sub(x[k], com);
        let w = sub(v[k], vcom);
        let dist = len(r);
        if dist < r_far_abs || dot(r, w) <= 0.0 {
            continue;
        }
        let mu = m[k] * mp / (m[k] + mp);
        let e_k = 0.5 * mu * dot(w, w) - m[k] * mp / dist;
        let dp = sub(x[j], x[i]);
        let dv = sub(v[j], v[i]);
        let e_pair = 0.5 * (m[i] * m[j] / mp) * dot(dv, dv) - m[i] * m[j] / len(dp);
        if e_k > 0.0 && e_pair < 0.0 {
            return Some(k);
        }
    }
    None
}

/// Builds the state for stars at rest at `x`, with the closest pair as the Jacobi pair.
pub fn from_rest(x: [V2; 3], m: &[f64; 3]) -> State {
    let mt: f64 = m.iter().sum();
    let com = mul(add(add(mul(x[0], m[0]), mul(x[1], m[1])), mul(x[2], m[2])), 1.0 / mt);
    let d = |a: usize, b: usize| len(sub(x[a], x[b]));
    // Third star = the one opposite the shortest side.
    let k = [(d(1, 2), 0), (d(0, 2), 1), (d(0, 1), 2)]
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap()
        .1;
    let (i, j) = pair(k);
    let mp = m[i] + m[j];
    let cp = mul(add(mul(x[i], m[i]), mul(x[j], m[j])), 1.0 / mp);
    State {
        k,
        q: sub(x[j], x[i]),
        u: [0.0; 2],
        qq: sub(x[k], cp),
        uu: [0.0; 2],
        t: 0.0,
        com,
    }
}

pub fn initial_state(cfg: &Config, c: V2) -> State {
    from_rest([cfg.pos_a, cfg.pos_b, c], &cfg.masses)
}

/// Runs one system to completion. `trace`, if given, receives (time, absolute positions)
/// after every step.
pub fn simulate_state(
    cfg: &Config,
    mut s: State,
    mut trace: Option<&mut Vec<(f64, [V2; 3])>>,
) -> Outcome {
    let m = &cfg.masses;
    let e0 = energy(&s, m);
    let (t_cr, r_v) = scales(m, e0);
    let t_max = cfg.t_max_cr * t_cr;
    let r_far_abs = cfg.r_far * r_v;
    let h = cfg.eta * e0.abs() * t_cr;
    let mut steps = 0u32;
    let mut fate = Fate::Undecided;
    while s.t < t_max && steps < cfg.max_steps {
        let h_eff = if cfg.resolve_binaries {
            h * (min_separation(&s, m) / r_v).sqrt().min(1.0)
        } else {
            h
        };
        step(&mut s, m, h_eff, -e0);
        maybe_switch(&mut s, m);
        steps += 1;
        if let Some(tr) = trace.as_deref_mut() {
            tr.push((s.t, absolute_positions(&s, m)));
        }
        if steps % ESCAPE_CHECK_EVERY == 0 {
            if let Some(k) = escaped(&s, m, r_far_abs) {
                fate = Fate::Ejected(k);
                break;
            }
        }
    }
    Outcome {
        fate,
        t_cr_units: s.t / t_cr,
        steps,
        energy_err: ((energy(&s, m) - e0) / e0).abs(),
    }
}

/// Runs the pixel experiment: stars 0 and 1 fixed, star 2 at `c`, all at rest.
pub fn simulate(cfg: &Config, c: V2) -> Outcome {
    simulate_state(cfg, initial_state(cfg, c), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Burrau's Pythagorean problem (Szebehely & Peters 1967): masses 3, 4, 5 at rest on a
    /// 3-4-5 triangle, each opposite the side of matching length. The lightest is ejected
    /// and the other two leave as a bound pair.
    #[test]
    fn pythagorean_problem_ejects_lightest() {
        let cfg = Config {
            masses: [3.0, 4.0, 5.0],
            eta: 0.002,
            t_max_cr: 1e4,
            max_steps: 20_000_000,
            ..Default::default()
        };
        let s = from_rest([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]], &cfg.masses);
        let out = simulate_state(&cfg, s, None);
        assert_eq!(out.fate, Fate::Ejected(0), "{out:?}");
        assert!(out.energy_err < 1e-6, "{out:?}");
    }

    #[test]
    fn coordinates_round_trip() {
        let m = [1.0, 2.0, 3.0];
        let x = [[0.3, -1.0], [1.2, 0.4], [-0.7, 0.9]];
        let s = from_rest(x, &m);
        for k in 0..3 {
            let t = with_third(&s, &m, k);
            let y = absolute_positions(&t, &m);
            for n in 0..3 {
                assert!(len(sub(y[n], x[n])) < 1e-12, "third {k} star {n}");
            }
            assert!((energy(&t, &m) - energy(&s, &m)).abs() < 1e-12);
        }
    }

    #[test]
    fn circular_binary_conserves_energy() {
        // Unit masses at separation 2 (circular relative speed sqrt(M / r) = 1), third star
        // far away and negligible.
        let m = [1.0, 1.0, 1e-9];
        let mut s = State {
            k: 2,
            q: [2.0, 0.0],
            u: [0.0, 1.0],
            qq: [1e4, 0.0],
            uu: [0.0, 0.0],
            t: 0.0,
            com: [0.0; 2],
        };
        let e0 = energy(&s, &m);
        let period = 2.0 * std::f64::consts::PI * 2.0;
        while s.t < 100.0 * period {
            step(&mut s, &m, 0.02, -e0);
        }
        let err = ((energy(&s, &m) - e0) / e0).abs();
        assert!(err < 1e-8, "relative energy error {err}");
        assert!((len(s.q) - 2.0).abs() < 1e-6, "radius {}", len(s.q));
    }

    #[test]
    fn mirror_symmetry() {
        let cfg = Config::default();
        for &(x, y) in &[(0.3, 0.9), (-1.7, 1.2), (0.05, 2.1), (1.4, 0.4)] {
            let a = simulate(&cfg, [x, y]);
            let b = simulate(&cfg, [x, -y]);
            if a.t_cr_units < 20.0 {
                assert_eq!(a.fate, b.fate, "({x},{y})");
            }
        }
    }

    // Off the symmetry axes: with star 2 exactly on x = 0 or y = 0, stars collide head-on.
    // Pixel centres never land exactly on those axes.
    #[test]
    fn generic_start_conserves_energy() {
        let out = simulate(&Config::default(), [0.3, 2.0]);
        assert!(out.steps > 0);
        assert!(out.energy_err < 1e-4, "{out:?}");
    }
}
