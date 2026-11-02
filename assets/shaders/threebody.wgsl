// Three-body ejection fractal.
//
// Every simulated pixel is its own three-body problem: stars 0 and 1 start at fixed points,
// star 2 starts at the pixel's position, all at rest. The pixel is coloured by which star
// is eventually ejected. Mirrors src/physics_cpu.rs (the f64 reference).
//
// Entry points, dispatched in this order by src/render/dispatch.rs:
//   reproject   - on a view change: warp the previous image into the new view as a preview
//   init        - on a view or level change: seed every pixel's initial state
//   advance     - every frame: advance each running pixel by up to `steps_per_frame` steps
//   clear_stats - every frame: zero the stats buffer
//   shade       - every frame: colour finished pixels, accumulate stats
//
// Scattering mode (params.mode == 1) reuses init/advance with a different setup: every
// thread is one random binary-single encounter for one cell of an outcome table, and
// starts a new random encounter as soon as one finishes. shade_scatter colours the cells.

struct Params {
    masses: vec4<f32>,
    center: vec2<f32>,
    prev_center: vec2<f32>,
    pos_a: vec2<f32>,
    pos_b: vec2<f32>,
    // World units per output texel.
    scale: f32,
    prev_scale: f32,
    eta: f32,
    eta_b: f32,
    t_max_cr: f32,
    r_far: f32,
    // Simulation resolution at the current refinement level (divides tex_size).
    res: u32,
    tex_size: u32,
    generation: u32,
    level: u32,
    steps_per_frame: u32,
    max_steps: u32,
    color_mode: u32,
    sensitivity: u32,
    front: u32,
    done: u32,
    // 0 = fate map, 1 = scattering table.
    mode: u32,
    // Binary eccentricity for scatterings; negative = thermal.
    scatter_ecc: f32,
    // Scattering table: cells per side, encounters running per cell, axis ranges.
    cells: u32,
    samples: u32,
    v_max: f32,
    rp_max: f32,
    seed_base: u32,
}

// 64 bytes per simulated system. The state is in Jacobi coordinates about the closest
// pair (see src/physics_cpu.rs): close separations are stored directly, which is what
// lets f32 get through near-collisions.
struct Pixel {
    q_u: vec4<f32>, // pair separation x_j - x_i, and its rate
    qq_uu: vec4<f32>, // third star relative to the pair's centre of mass, and its rate
    t: f32,
    t_max: f32,
    e0: f32,
    // bits 0-1 status (0 running, 1 finished, 2 undecided), bits 2-3 fate (0-2 = star
    // ejected, 3 = ionised), bits 4-5 third star k, bits 6.. steps
    info: u32,
    // Scattering mode: encounter counter (random seed) and initial binary energy.
    trial: u32,
    e_b0: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
// Lane 0 is the normal run; lane 1 (sensitivity mode) reruns every pixel with eta_b.
@group(0) @binding(1) var<storage, read_write> states: array<Pixel>;
// Header (see STATS_* below) followed by a 64x64 grid of [fate0, fate1, fate2, undecided].
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;
// Fate mode: per-pixel `info` of lane 0 at the current level, for GPU-vs-CPU checks.
// Scattering mode: ACC_STRIDE counters per table cell (see ACC_*).
@group(0) @binding(3) var<storage, read_write> fates: array<atomic<u32>>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var src: texture_2d<f32>;

const ESCAPE_CHECK_EVERY: u32 = 4u;
const SWITCH_RATIO: f32 = 0.8;
const STATUS_RUNNING: u32 = 0u;
const STATUS_EJECTED: u32 = 1u;
const STATUS_UNDECIDED: u32 = 2u;
const MAX_STEPS_FIELD: u32 = 0x03ffffffu;
const FATE_IONIZED: u32 = 3u;
const NO_ESCAPE: u32 = 4u;

const MODE_SCATTER: u32 = 1u;
const ACC_STRIDE: u32 = 8u;
const ACC_TRIALS: u32 = 0u;
const ACC_FLYBY: u32 = 1u;
const ACC_EXCHANGE: u32 = 2u;
const ACC_IONIZED: u32 = 3u;
const ACC_UNDECIDED: u32 = 4u;
const ACC_DE_SUM: u32 = 5u; // i32, fractional binding-energy change x DE_SCALE
const ACC_DE_COUNT: u32 = 6u;
const DE_SCALE: f32 = 1000.0;
// Must match SCATTER_R0 / SCATTER_RESONANCE_TIME in src/physics_cpu.rs.
const SCATTER_R0: f32 = 20.0;
const SCATTER_RESONANCE_TIME: f32 = 2000.0;
const TAU: f32 = 6.283185307179586;

const STATS_GENERATION: u32 = 0u;
const STATS_LEVEL: u32 = 1u;
const STATS_COUNTS: u32 = 2u; // fate0, fate1, fate2, undecided, running, disagree
const STATS_HEADER: u32 = 16u;
const GRID: u32 = 64u;

// Yoshida (1990) 4th-order triple-jump weights.
const YOSHIDA_W1: f32 = 1.3512071919596576;
const YOSHIDA_W0: f32 = -1.7024143839193153;

// ---------------------------------------------------------------------------------------
// Physics (mirrors src/physics_cpu.rs)

struct Jac {
    q: vec2<f32>,
    u: vec2<f32>,
    qq: vec2<f32>,
    uu: vec2<f32>,
    t: f32,
    k: u32,
}

struct Seps {
    ik: vec2<f32>, // x_k - x_i
    jk: vec2<f32>, // x_k - x_j
}

fn mass(n: u32) -> f32 {
    return params.masses[n];
}

fn total_mass() -> f32 {
    return params.masses.x + params.masses.y + params.masses.z;
}

fn third_separations(s: Jac) -> Seps {
    let i = (s.k + 1u) % 3u;
    let j = (s.k + 2u) % 3u;
    let mp = mass(i) + mass(j);
    return Seps(s.qq + s.q * (mass(j) / mp), s.qq - s.q * (mass(i) / mp));
}

fn third_rates(s: Jac) -> Seps {
    let i = (s.k + 1u) % 3u;
    let j = (s.k + 2u) % 3u;
    let mp = mass(i) + mass(j);
    return Seps(s.uu + s.u * (mass(j) / mp), s.uu - s.u * (mass(i) / mp));
}

// U = sum m_i m_j / r_ij (positive).
fn potential(s: Jac) -> f32 {
    let i = (s.k + 1u) % 3u;
    let j = (s.k + 2u) % 3u;
    let r = third_separations(s);
    return mass(i) * mass(j) / length(s.q) + mass(i) * mass(s.k) / length(r.ik)
        + mass(j) * mass(s.k) / length(r.jk);
}

fn kinetic(s: Jac) -> f32 {
    let i = (s.k + 1u) % 3u;
    let j = (s.k + 2u) % 3u;
    let mp = mass(i) + mass(j);
    let mt = mp + mass(s.k);
    return 0.5 * (mass(i) * mass(j) / mp) * dot(s.u, s.u)
        + 0.5 * (mass(s.k) * mp / mt) * dot(s.uu, s.uu);
}

fn energy(s: Jac) -> f32 {
    return kinetic(s) - potential(s);
}

fn inv_cube(r: vec2<f32>) -> vec2<f32> {
    let r2 = dot(r, r);
    return r / (r2 * sqrt(r2));
}

// One logH leapfrog step of length h in regularised time; b = -E0.
fn leapfrog(s: ptr<function, Jac>, h: f32, b: f32) {
    var dt = 0.5 * h / (kinetic(*s) + b);
    (*s).q += (*s).u * dt;
    (*s).qq += (*s).uu * dt;
    (*s).t += dt;

    let k = (*s).k;
    let i = (k + 1u) % 3u;
    let j = (k + 2u) % 3u;
    let mp = mass(i) + mass(j);
    let mt = mp + mass(k);
    let r = third_separations(*s);
    let fq = inv_cube((*s).q);
    let fik = inv_cube(r.ik);
    let fjk = inv_cube(r.jk);
    dt = h / potential(*s);
    (*s).u += (-mp * fq + mass(k) * (fjk - fik)) * dt;
    (*s).uu += (-(mt / mp) * (mass(i) * fik + mass(j) * fjk)) * dt;

    dt = 0.5 * h / (kinetic(*s) + b);
    (*s).q += (*s).u * dt;
    (*s).qq += (*s).uu * dt;
    (*s).t += dt;
}

fn yoshida(s: ptr<function, Jac>, h: f32, b: f32) {
    leapfrog(s, YOSHIDA_W1 * h, b);
    leapfrog(s, YOSHIDA_W0 * h, b);
    leapfrog(s, YOSHIDA_W1 * h, b);
}

// Re-expresses the state with star c as the third, from separations only.
fn with_third(s: Jac, c: u32) -> Jac {
    let i = (s.k + 1u) % 3u;
    let j = (s.k + 2u) % 3u;
    let rs = third_separations(s);
    let ws = third_rates(s);
    // r[n] = x_b - x_a with (a, b) = pair(n).
    var r: array<vec2<f32>, 3>;
    var w: array<vec2<f32>, 3>;
    r[s.k] = s.q;
    w[s.k] = s.u;
    r[i] = rs.jk;
    w[i] = ws.jk;
    r[j] = -rs.ik;
    w[j] = -ws.ik;
    let a = (c + 1u) % 3u;
    let b = (c + 2u) % 3u;
    let mab = mass(a) + mass(b);
    var out: Jac;
    out.k = c;
    out.t = s.t;
    out.q = r[c];
    out.u = w[c];
    out.qq = (r[a] * mass(b) - r[b] * mass(a)) / mab;
    out.uu = (w[a] * mass(b) - w[b] * mass(a)) / mab;
    return out;
}

fn maybe_switch(s: ptr<function, Jac>) {
    let i = ((*s).k + 1u) % 3u;
    let j = ((*s).k + 2u) % 3u;
    let r = third_separations(*s);
    let dq = length((*s).q);
    let dik = length(r.ik);
    let djk = length(r.jk);
    if dik < SWITCH_RATIO * dq && dik <= djk {
        *s = with_third(*s, j);
    } else if djk < SWITCH_RATIO * dq {
        *s = with_third(*s, i);
    }
}

fn min_separation(s: Jac) -> f32 {
    let r = third_separations(s);
    return min(length(s.q), min(length(r.ik), length(r.jk)));
}

// total mass^(5/2) / (2|E|)^(3/2)
fn crossing_time(e0: f32) -> f32 {
    return pow(total_mass(), 2.5) / pow(2.0 * abs(e0), 1.5);
}

fn virial_radius(e0: f32) -> f32 {
    let m = params.masses;
    return (m.x * m.y + m.x * m.z + m.y * m.z) / abs(e0);
}

fn two_body_energy(x: array<vec2<f32>, 3>, v: array<vec2<f32>, 3>, a: u32, b: u32) -> f32 {
    let ma = mass(a);
    let mb = mass(b);
    let dv = v[b] - v[a];
    return 0.5 * (ma * mb / (ma + mb)) * dot(dv, dv) - ma * mb / length(x[b] - x[a]);
}

// A star beyond r_far_abs, receding, with positive energy relative to the other two:
// returns its index if the pair left behind is bound (writing that pair's energy to
// e_pair_out), FATE_IONIZED once every pair is unbound and far apart, else NO_ESCAPE.
fn escaped(s: Jac, r_far_abs: f32, e_pair_out: ptr<function, f32>) -> u32 {
    let i0 = (s.k + 1u) % 3u;
    let j0 = (s.k + 2u) % 3u;
    let mp0 = mass(i0) + mass(j0);
    let mt = mp0 + mass(s.k);
    var x: array<vec2<f32>, 3>;
    var v: array<vec2<f32>, 3>;
    let cp = s.qq * (-mass(s.k) / mt);
    let cv = s.uu * (-mass(s.k) / mt);
    x[i0] = cp - s.q * (mass(j0) / mp0);
    x[j0] = cp + s.q * (mass(i0) / mp0);
    x[s.k] = s.qq * (mp0 / mt);
    v[i0] = cv - s.u * (mass(j0) / mp0);
    v[j0] = cv + s.u * (mass(i0) / mp0);
    v[s.k] = s.uu * (mp0 / mt);
    for (var k = 0u; k < 3u; k++) {
        let i = (k + 1u) % 3u;
        let j = (k + 2u) % 3u;
        let mi = mass(i);
        let mj = mass(j);
        let mk = mass(k);
        let mp = mi + mj;
        let r = x[k] - (mi * x[i] + mj * x[j]) / mp;
        let w = v[k] - (mi * v[i] + mj * v[j]) / mp;
        let dist = length(r);
        if dist < r_far_abs || dot(r, w) <= 0.0 {
            continue;
        }
        let e_k = 0.5 * (mk * mp / (mk + mp)) * dot(w, w) - mk * mp / dist;
        if e_k <= 0.0 {
            continue;
        }
        let e_pair = two_body_energy(x, v, i, j);
        if e_pair < 0.0 {
            *e_pair_out = e_pair;
            return k;
        }
        if length(x[j] - x[i]) > r_far_abs && two_body_energy(x, v, k, i) >= 0.0
            && two_body_energy(x, v, k, j) >= 0.0 {
            return FATE_IONIZED;
        }
    }
    return NO_ESCAPE;
}

fn load(p: Pixel) -> Jac {
    return Jac(p.q_u.xy, p.q_u.zw, p.qq_uu.xy, p.qq_uu.zw, p.t, (p.info >> 4u) & 3u);
}

// World position of simulated pixel `p` at resolution `res`, sampled at its block centre.
fn pixel_world(p: vec2<u32>, res: u32) -> vec2<f32> {
    let tex = f32(params.tex_size);
    let block = tex / f32(res);
    let uv = (vec2<f32>(p) + 0.5) * block - 0.5 * tex;
    return params.center + vec2<f32>(uv.x, -uv.y) * params.scale;
}

fn lane_count() -> u32 {
    return select(1u, 2u, params.sensitivity != 0u);
}

// ---------------------------------------------------------------------------------------
// Setups

struct Scales {
    h: f32,
    r_ref: f32,
    r_far: f32,
}

// Mirrors fate_scales / scatter_state in src/physics_cpu.rs.
fn run_scales(p: Pixel, eta: f32) -> Scales {
    if params.mode == MODE_SCATTER {
        let mb = mass(0u) + mass(1u);
        return Scales(eta * mass(0u) * mass(1u) * sqrt(1.0 / mb), 1.0, SCATTER_R0);
    }
    let t_cr = p.t_max / params.t_max_cr;
    let r_v = virial_radius(p.e0);
    return Scales(eta * abs(p.e0) * t_cr, r_v, params.r_far * r_v);
}

fn store_jac(p: ptr<function, Pixel>, s: Jac) {
    (*p).q_u = vec4<f32>(s.q, s.u);
    (*p).qq_uu = vec4<f32>(s.qq, s.uu);
    (*p).t = s.t;
}

fn init_fate(id: vec2<u32>, res: u32) -> Pixel {
    var x: array<vec2<f32>, 3>;
    x[0] = params.pos_a;
    x[1] = params.pos_b;
    x[2] = pixel_world(id, res);
    // Third star = the one opposite the shortest side.
    let d0 = length(x[2] - x[1]);
    let d1 = length(x[2] - x[0]);
    let d2 = length(x[1] - x[0]);
    var k = 2u;
    if d0 <= d1 && d0 <= d2 {
        k = 0u;
    } else if d1 <= d2 {
        k = 1u;
    }
    let i = (k + 1u) % 3u;
    let j = (k + 2u) % 3u;
    let mp = mass(i) + mass(j);
    var s: Jac;
    s.k = k;
    s.t = 0.0;
    s.q = x[j] - x[i];
    s.u = vec2<f32>(0.0);
    s.qq = x[k] - (mass(i) * x[i] + mass(j) * x[j]) / mp;
    s.uu = vec2<f32>(0.0);
    let e0 = -potential(s);

    var p: Pixel;
    store_jac(&p, s);
    p.t_max = params.t_max_cr * crossing_time(e0);
    p.e0 = e0;
    p.info = (k << 4u) | STATUS_RUNNING;
    return p;
}

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn unit_float(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

fn critical_speed() -> f32 {
    let mb = mass(0u) + mass(1u);
    let mu3 = mass(2u) * mb / (mb + mass(2u));
    return sqrt(mass(0u) * mass(1u) / mu3);
}

// Table cell of system `idx`: x = v_inf / v_c (left to right), y = r_p / a (bottom to top).
fn scatter_cell(idx: u32) -> vec2<u32> {
    let cell = idx / params.samples;
    return vec2<u32>(cell % params.cells, cell / params.cells);
}

fn cell_params(c: vec2<u32>) -> vec2<f32> {
    let n = f32(params.cells);
    return vec2<f32>(
        (f32(c.x) + 0.5) / n * params.v_max,
        (n - f32(c.y) - 0.5) / n * params.rp_max,
    );
}

// One random binary-single encounter. Mirrors scatter_state in src/physics_cpu.rs.
fn init_scatter(idx: u32, trial: u32) -> Pixel {
    let vr = cell_params(scatter_cell(idx));
    let m0 = mass(0u);
    let m1 = mass(1u);
    let mb = m0 + m1;
    let mt = mb + mass(2u);

    var h = pcg(idx ^ pcg(trial + params.seed_base));
    let r_mean = unit_float(h);
    h = pcg(h);
    let r_orient = unit_float(h);
    h = pcg(h);
    let r_sense = unit_float(h);
    h = pcg(h);
    let r_ecc = unit_float(h);

    var e = params.scatter_ecc;
    if e < 0.0 {
        e = sqrt(r_ecc);
    }
    e = min(e, 0.99);
    let mean = TAU * r_mean;
    var ea = mean;
    for (var n = 0; n < 12; n++) {
        ea -= (ea - e * sin(ea) - mean) / (1.0 - e * cos(ea));
    }
    let f = sqrt(mb) / (1.0 - e * cos(ea));
    let root = sqrt(1.0 - e * e);
    let sense = select(-1.0, 1.0, r_sense < 0.5);
    let pos = vec2<f32>(cos(ea) - e, sense * root * sin(ea));
    let vel = vec2<f32>(-sin(ea) * f, sense * root * cos(ea) * f);
    let so = sin(TAU * r_orient);
    let co = cos(TAU * r_orient);
    let rot = mat2x2<f32>(co, so, -so, co);

    let v_inf = max(vr.x * critical_speed(), 1e-3);
    let rp = vr.y;
    let b = rp * sqrt(1.0 + 2.0 * mt / (rp * v_inf * v_inf));
    let v_t = b * v_inf / SCATTER_R0;
    let v_r = -sqrt(max(v_inf * v_inf + 2.0 * mt / SCATTER_R0 - v_t * v_t, 0.0));

    var s: Jac;
    s.k = 2u;
    s.t = 0.0;
    s.q = rot * pos;
    s.u = rot * vel;
    s.qq = vec2<f32>(-SCATTER_R0, 0.0);
    s.uu = vec2<f32>(-v_r, -v_t);

    let t_b = sqrt(1.0 / mb);
    let t_ff = 0.25 * TAU * sqrt(SCATTER_R0 * SCATTER_R0 * SCATTER_R0 / (2.0 * mt));
    var p: Pixel;
    store_jac(&p, s);
    p.t_max = 4.0 * t_ff + SCATTER_RESONANCE_TIME * t_b;
    p.e0 = energy(s);
    p.e_b0 = 0.5 * (m0 * m1 / mb) * dot(s.u, s.u) - m0 * m1 / length(s.q);
    p.trial = trial;
    p.info = (2u << 4u) | STATUS_RUNNING;
    return p;
}

// Adds a finished encounter to its cell's counters.
fn accumulate(idx: u32, status: u32, fate: u32, de: f32) {
    let c = scatter_cell(idx);
    let base = (c.y * params.cells + c.x) * ACC_STRIDE;
    atomicAdd(&fates[base + ACC_TRIALS], 1u);
    if status == STATUS_UNDECIDED {
        atomicAdd(&fates[base + ACC_UNDECIDED], 1u);
    } else if fate == FATE_IONIZED {
        atomicAdd(&fates[base + ACC_IONIZED], 1u);
    } else {
        // Star 2 came in: if it leaves again it's a flyby, otherwise an exchange.
        atomicAdd(&fates[base + select(ACC_EXCHANGE, ACC_FLYBY, fate == 2u)], 1u);
        let fixed = i32(round(clamp(de, -20.0, 20.0) * DE_SCALE));
        atomicAdd(&fates[base + ACC_DE_SUM], bitcast<u32>(fixed));
        atomicAdd(&fates[base + ACC_DE_COUNT], 1u);
    }
}

// ---------------------------------------------------------------------------------------
// Entry points

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = params.res;
    if id.x >= res || id.y >= res || id.z >= lane_count() {
        return;
    }
    let idx = id.z * res * res + id.y * res + id.x;
    if params.mode == MODE_SCATTER {
        // Clear the counters; slot 7 of cell 0 tags them with the generation so stale
        // readbacks can be told apart.
        if idx < params.cells * params.cells * ACC_STRIDE {
            atomicStore(&fates[idx], select(0u, params.generation, idx == 7u));
        }
        states[idx] = init_scatter(idx, 0u);
    } else {
        states[idx] = init_fate(id.xy, res);
    }
}

@compute @workgroup_size(8, 8, 1)
fn advance(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = params.res;
    if id.x >= res || id.y >= res || id.z >= lane_count() {
        return;
    }
    let idx = id.z * res * res + id.y * res + id.x;
    var p = states[idx];
    if (p.info & 3u) != STATUS_RUNNING {
        return;
    }
    let eta = select(params.eta, params.eta_b, id.z == 1u);
    let sc = run_scales(p, eta);
    let b = -p.e0;

    var s = load(p);
    var steps = p.info >> 6u;
    var status = STATUS_RUNNING;
    var fate = 0u;
    var e_pair = 0.0;
    for (var n = 0u; n < params.steps_per_frame; n++) {
        // Shrink the step for tight pairs so they keep a fixed number of steps per orbit.
        let h_eff = sc.h * min(1.0, sqrt(min_separation(s) / sc.r_ref));
        yoshida(&s, h_eff, b);
        maybe_switch(&s);
        steps += 1u;
        if steps % ESCAPE_CHECK_EVERY == 0u {
            let e = escaped(s, sc.r_far, &e_pair);
            if e != NO_ESCAPE {
                status = STATUS_EJECTED;
                fate = e;
                break;
            }
        }
        if s.t >= p.t_max || steps >= params.max_steps {
            status = STATUS_UNDECIDED;
            break;
        }
    }

    if params.mode == MODE_SCATTER && status != STATUS_RUNNING {
        accumulate(idx, status, fate, (p.e_b0 - e_pair) / abs(p.e_b0));
        states[idx] = init_scatter(idx, p.trial + 1u);
        return;
    }
    store_jac(&p, s);
    p.info = (min(steps, MAX_STEPS_FIELD) << 6u) | (s.k << 4u) | (fate << 2u) | status;
    states[idx] = p;
}

@compute @workgroup_size(64, 1, 1)
fn clear_stats(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&stats) {
        return;
    }
    var value = 0u;
    if id.x == STATS_GENERATION {
        value = params.generation;
    } else if id.x == STATS_LEVEL {
        value = params.level;
    }
    atomicStore(&stats[id.x], value);
}

// ---------------------------------------------------------------------------------------
// Colour

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

fn fate_color(fate: u32) -> vec3<f32> {
    switch fate {
        case 0u: { return srgb_to_linear(vec3<f32>(1.0, 0.77, 0.20)); } // gold
        case 1u: { return srgb_to_linear(vec3<f32>(0.22, 0.52, 1.0)); } // blue
        default: { return srgb_to_linear(vec3<f32>(0.93, 0.22, 0.24)); } // red
    }
}

// Viridis, polynomial fit (Matt Zucker, CC0).
fn viridis(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    let c0 = vec3<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    let c1 = vec3<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    let c2 = vec3<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    let c3 = vec3<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    let c4 = vec3<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105);
    let c5 = vec3<f32>(4.776384997670288, -13.74514537774601, -65.35303263337234);
    let c6 = vec3<f32>(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    let srgb = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
    return srgb_to_linear(clamp(srgb, vec3<f32>(0.0), vec3<f32>(1.0)));
}

const UNDECIDED: vec3<f32> = vec3<f32>(0.006, 0.006, 0.008);

fn pixel_color(p: Pixel, disagree: bool) -> vec3<f32> {
    let status = p.info & 3u;
    let fate = (p.info >> 2u) & 3u;
    let t_cr = p.t / (p.t_max / params.t_max_cr);
    let t_frac = log(1.0 + t_cr) / log(1.0 + params.t_max_cr);
    switch params.color_mode {
        // Ejection time.
        case 1u: {
            if status == STATUS_UNDECIDED { return UNDECIDED; }
            return viridis(t_frac);
        }
        // Energy error, log10 from 1e-8 (dark) to 1e-1 (bright).
        case 2u: {
            let e = energy(load(p));
            let err = abs((e - p.e0) / p.e0);
            return viridis((log(max(err, 1e-10)) / log(10.0) + 8.0) / 7.0);
        }
        // Step count, log scale.
        case 3u: {
            let steps = f32(p.info >> 6u);
            return viridis(log(1.0 + steps) / log(1.0 + f32(params.max_steps)));
        }
        // Sensitivity: white where the accurate and less-accurate runs disagree.
        case 4u: {
            if disagree { return vec3<f32>(1.0); }
            if status == STATUS_UNDECIDED { return UNDECIDED; }
            return fate_color(fate) * 0.12;
        }
        // Fate, darker for later ejections.
        default: {
            if status == STATUS_UNDECIDED { return UNDECIDED; }
            return fate_color(fate) * (1.0 - 0.8 * pow(t_frac, 0.8));
        }
    }
}

var<workgroup> wg_counts: array<atomic<u32>, 6>;

@compute @workgroup_size(8, 8, 1)
fn shade(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if li < 6u {
        atomicStore(&wg_counts[li], 0u);
    }
    workgroupBarrier();

    let res = params.res;
    if id.x < res && id.y < res {
        let idx = id.y * res + id.x;
        let p = states[idx];
        let status = p.info & 3u;
        let fate = (p.info >> 2u) & 3u;
        atomicStore(&fates[idx], p.info);

        var running = status == STATUS_RUNNING;
        var disagree = false;
        if params.sensitivity != 0u {
            let q = states[res * res + idx];
            let q_status = q.info & 3u;
            running = running || q_status == STATUS_RUNNING;
            // Compare outcome category: fate 0/1/2 or undecided.
            let a = select(3u, fate, status == STATUS_EJECTED);
            let b = select(3u, (q.info >> 2u) & 3u, q_status == STATUS_EJECTED);
            disagree = !running && a != b;
        }

        if running {
            atomicAdd(&wg_counts[4], 1u);
        } else {
            let k = select(3u, fate, status == STATUS_EJECTED);
            atomicAdd(&wg_counts[k], 1u);
            if disagree {
                atomicAdd(&wg_counts[5], 1u);
            }
            let cell = (id.y * GRID / res) * GRID + id.x * GRID / res;
            atomicAdd(&stats[STATS_HEADER + cell * 4u + k], 1u);
        }

        // Lane 0 finished: paint its block (sensitivity mode waits for both lanes).
        if !running {
            let color = vec4<f32>(pixel_color(p, disagree), 1.0);
            let block = params.tex_size / res;
            for (var dy = 0u; dy < block; dy++) {
                for (var dx = 0u; dx < block; dx++) {
                    textureStore(dst, vec2<u32>(id.x * block + dx, id.y * block + dy), color);
                }
            }
        }
    }

    workgroupBarrier();
    if li < 6u {
        atomicAdd(&stats[STATS_COUNTS + li], atomicLoad(&wg_counts[li]));
    }
}

// Colours each table cell from its counters: outcome mix (blue flyby, gold exchange,
// red ionisation, dark unresolved) or, in colour mode 1, the mean change of the binary's
// binding energy (green = hardened, magenta = softened).
@compute @workgroup_size(8, 8, 1)
fn shade_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = params.cells;
    if id.x >= n || id.y >= n {
        return;
    }
    let base = (id.y * n + id.x) * ACC_STRIDE;
    let trials = atomicLoad(&fates[base + ACC_TRIALS]);
    var color = vec3<f32>(0.02);
    if trials > 0u {
        let inv = 1.0 / f32(trials);
        if params.color_mode == 1u {
            let count = atomicLoad(&fates[base + ACC_DE_COUNT]);
            if count > 0u {
                let mean = f32(bitcast<i32>(atomicLoad(&fates[base + ACC_DE_SUM])))
                    / DE_SCALE / f32(count);
                let t = pow(clamp(abs(mean) / 0.5, 0.0, 1.0), 0.6);
                let hue = select(vec3<f32>(0.85, 0.2, 0.85), vec3<f32>(0.25, 0.9, 0.35), mean > 0.0);
                color = srgb_to_linear(mix(vec3<f32>(0.08), hue, t));
            }
        } else {
            let fly = f32(atomicLoad(&fates[base + ACC_FLYBY])) * inv;
            let exch = f32(atomicLoad(&fates[base + ACC_EXCHANGE])) * inv;
            let ion = f32(atomicLoad(&fates[base + ACC_IONIZED])) * inv;
            color = fly * fate_color(1u) + exch * fate_color(0u) + ion * fate_color(2u)
                + (1.0 - fly - exch - ion) * UNDECIDED;
        }
    }
    let block = params.tex_size / n;
    for (var dy = 0u; dy < block; dy++) {
        for (var dx = 0u; dx < block; dx++) {
            textureStore(dst, vec2<u32>(id.x * block + dx, id.y * block + dy), vec4<f32>(color, 1.0));
        }
    }
}

// Warp the previous view's image into the new view so zooming and panning feel instant
// while the new pixels are computed.
@compute @workgroup_size(8, 8, 1)
fn reproject(@builtin(global_invocation_id) id: vec3<u32>) {
    let tex = params.tex_size;
    if id.x >= tex || id.y >= tex {
        return;
    }
    let half = 0.5 * f32(tex);
    let uv = vec2<f32>(id.xy) + 0.5 - half;
    let world = params.center + vec2<f32>(uv.x, -uv.y) * params.scale;
    let rel = (world - params.prev_center) / params.prev_scale;
    let old = vec2<f32>(rel.x, -rel.y) + half - 0.5;

    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if all(old >= vec2<f32>(-0.5)) && all(old <= vec2<f32>(f32(tex) - 0.5)) {
        // Manual bilinear filter (textureLoad has no sampler).
        let base = floor(old);
        let f = old - base;
        let maxc = vec2<i32>(i32(tex) - 1);
        let p00 = clamp(vec2<i32>(base), vec2<i32>(0), maxc);
        let p11 = clamp(vec2<i32>(base) + 1, vec2<i32>(0), maxc);
        let c00 = textureLoad(src, p00, 0);
        let c10 = textureLoad(src, vec2<i32>(p11.x, p00.y), 0);
        let c01 = textureLoad(src, vec2<i32>(p00.x, p11.y), 0);
        let c11 = textureLoad(src, p11, 0);
        color = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    }
    textureStore(dst, id.xy, color);
}
