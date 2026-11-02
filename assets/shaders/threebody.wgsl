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
}

// 48 bytes per simulated system. The state is in Jacobi coordinates about the closest
// pair (see src/physics_cpu.rs): close separations are stored directly, which is what
// lets f32 get through near-collisions.
struct Pixel {
    q_u: vec4<f32>, // pair separation x_j - x_i, and its rate
    qq_uu: vec4<f32>, // third star relative to the pair's centre of mass, and its rate
    t: f32,
    t_max: f32,
    e0: f32,
    // bits 0-1 status (0 running, 1 ejected, 2 undecided), bits 2-3 fate,
    // bits 4-5 third star k, bits 6.. steps
    info: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Lane 0 is the normal run; lane 1 (sensitivity mode) reruns every pixel with eta_b.
@group(0) @binding(1) var<storage, read_write> states: array<Pixel>;
// Header (see STATS_* below) followed by a 64x64 grid of [fate0, fate1, fate2, undecided].
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;
// Per-pixel `info` of lane 0 at the current level, for GPU-vs-CPU verification.
@group(0) @binding(3) var<storage, read_write> fates: array<u32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var src: texture_2d<f32>;

const ESCAPE_CHECK_EVERY: u32 = 4u;
const SWITCH_RATIO: f32 = 0.8;
const STATUS_RUNNING: u32 = 0u;
const STATUS_EJECTED: u32 = 1u;
const STATUS_UNDECIDED: u32 = 2u;
const MAX_STEPS_FIELD: u32 = 0x03ffffffu;

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

// Index of the escaping star, or 3 if none: positive energy relative to the other two,
// moving away, beyond r_far_abs, and leaving a bound pair behind.
fn escaped(s: Jac, r_far_abs: f32) -> u32 {
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
        let dv = v[j] - v[i];
        let e_pair = 0.5 * (mi * mj / mp) * dot(dv, dv) - mi * mj / length(x[j] - x[i]);
        if e_k > 0.0 && e_pair < 0.0 {
            return k;
        }
    }
    return 3u;
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
// Entry points

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = params.res;
    if id.x >= res || id.y >= res || id.z >= lane_count() {
        return;
    }
    var x: array<vec2<f32>, 3>;
    x[0] = params.pos_a;
    x[1] = params.pos_b;
    x[2] = pixel_world(id.xy, res);
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
    p.q_u = vec4<f32>(s.q, s.u);
    p.qq_uu = vec4<f32>(s.qq, s.uu);
    p.t = 0.0;
    p.t_max = params.t_max_cr * crossing_time(e0);
    p.e0 = e0;
    p.info = (k << 4u) | STATUS_RUNNING;
    states[id.z * res * res + id.y * res + id.x] = p;
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
    let t_cr = p.t_max / params.t_max_cr;
    let r_v = virial_radius(p.e0);
    let r_far_abs = params.r_far * r_v;
    let h = eta * abs(p.e0) * t_cr;
    let b = -p.e0;

    var s = load(p);
    var steps = p.info >> 6u;
    var status = STATUS_RUNNING;
    var fate = 0u;
    for (var n = 0u; n < params.steps_per_frame; n++) {
        // Shrink the step for tight pairs so they keep a fixed number of steps per orbit.
        let h_eff = h * min(1.0, sqrt(min_separation(s) / r_v));
        yoshida(&s, h_eff, b);
        maybe_switch(&s);
        steps += 1u;
        if steps % ESCAPE_CHECK_EVERY == 0u {
            let e = escaped(s, r_far_abs);
            if e < 3u {
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

    p.q_u = vec4<f32>(s.q, s.u);
    p.qq_uu = vec4<f32>(s.qq, s.uu);
    p.t = s.t;
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
        fates[idx] = p.info;

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
