//! Prints the run of a single starting point: cargo run --release --example probe -- x y
use bevy_threebody::physics_cpu::{Config, energy, initial_state, simulate_state};

fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let cfg = Config::default();
    let s = initial_state(&cfg, [a[0], a[1]]);
    let e0 = energy(&s, &cfg.masses);
    let mut tr = Vec::new();
    let out = simulate_state(&cfg, s, Some(&mut tr));
    println!("{out:?} e0 {e0}");
    let mut minr: f64 = 1e9;
    let mut prev_t = 0.0;
    for (n, (t, x)) in tr.iter().enumerate() {
        let r = |i: usize, j: usize| ((x[i][0] - x[j][0]).powi(2) + (x[i][1] - x[j][1]).powi(2)).sqrt();
        let rm = r(0, 1).min(r(0, 2)).min(r(1, 2));
        if rm < minr {
            minr = rm;
            println!("step {n} t {t:.6} dt {:.3e} min r {rm:.3e}", t - prev_t);
        }
        prev_t = *t;
    }
}
