//! CPU reference for the scattering table: outcome probabilities for a few cells.
//!
//! cargo run --release --example scatter -- [trials] [ecc]

use bevy_threebody::physics_cpu::{Config, Encounter, Fate, simulate_encounter};

fn main() {
    let args: Vec<f64> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let trials = args.first().copied().unwrap_or(400.0) as usize;
    let ecc = args.get(1).copied().unwrap_or(0.0);
    let cfg = Config::default();
    println!("v/vc   rp/a  flyby  exch   ion    undec  <dE/E>   (n = {trials}, e = {ecc})");
    for &v_ratio in &[0.1, 0.5, 1.0, 2.0] {
        for &rp in &[0.25, 1.0, 2.5] {
            let enc = Encounter { v_ratio, rp, ecc };
            let results: Vec<_> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..trials)
                    .map(|n| {
                        sc.spawn(move || {
                            let mut seed = (n as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
                            let r = std::array::from_fn(|_| {
                                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                (seed >> 11) as f64 / (1u64 << 53) as f64
                            });
                            simulate_encounter(&cfg, &enc, r)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let frac = |f: &dyn Fn(Fate) -> bool| {
                100.0 * results.iter().filter(|(o, _)| f(o.fate)).count() as f64 / trials as f64
            };
            let des: Vec<f64> = results.iter().map(|r| r.1).filter(|d| d.is_finite()).collect();
            println!(
                "{v_ratio:<6} {rp:<5} {:5.1}% {:5.1}% {:5.1}% {:5.1}% {:+.3}",
                frac(&|f| f == Fate::Ejected(2)),
                frac(&|f| matches!(f, Fate::Ejected(0) | Fate::Ejected(1))),
                frac(&|f| f == Fate::Ionized),
                frac(&|f| f == Fate::Undecided),
                des.iter().sum::<f64>() / des.len().max(1) as f64
            );
        }
    }
}
