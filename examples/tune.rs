//! Runs the CPU reference over a coarse grid of the default view and prints fate
//! statistics, step counts and timing. Used to tune eta / t_max / r_far.
//!
//! cargo run --release --example tune -- [res] [eta] [t_max_cr] [r_far] [half_width]
//!     [resolve_binaries 0|1] [max_steps]

use bevy_threebody::physics_cpu::{Config, Fate, simulate};
use std::time::Instant;

fn main() {
    let args: Vec<f64> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let res = args.first().copied().unwrap_or(96.0) as usize;
    let mut cfg = Config::default();
    if let Some(&e) = args.get(1) { cfg.eta = e; }
    if let Some(&t) = args.get(2) { cfg.t_max_cr = t; }
    if let Some(&r) = args.get(3) { cfg.r_far = r; }
    let half = args.get(4).copied().unwrap_or(3.0);
    if let Some(&r) = args.get(5) { cfg.resolve_binaries = r != 0.0; }
    if let Some(&r) = args.get(6) { cfg.max_steps = r as u32; }

    let start = Instant::now();
    let rows: Vec<Vec<_>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..res)
            .map(|j| {
                sc.spawn(move || {
                    (0..res)
                        .map(|i| {
                            let x = -half + (i as f64 + 0.5) / res as f64 * 2.0 * half;
                            let y = half - (j as f64 + 0.5) / res as f64 * 2.0 * half;
                            let mut fine = cfg;
                            fine.eta /= 2.0;
                            (simulate(&cfg, [x, y]), simulate(&fine, [x, y]).fate)
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let pairs: Vec<_> = rows.iter().flatten().collect();
    let all: Vec<_> = pairs.iter().map(|(o, _)| o).collect();
    let (mut early, mut early_ok, mut late, mut late_ok) = (0, 0, 0, 0);
    for (o, fine) in &pairs {
        let ok = o.fate == *fine;
        if matches!(o.fate, Fate::Ejected(_)) && o.t_cr_units < 10.0 {
            early += 1;
            early_ok += ok as usize;
        } else {
            late += 1;
            late_ok += ok as usize;
        }
    }
    let rows: Vec<Vec<_>> = rows.iter().map(|r| r.iter().map(|(o, _)| *o).collect()).collect();
    let n = all.len() as f64;
    let mut counts = [0usize; 4];
    let mut steps: Vec<u32> = all.iter().map(|o| o.steps).collect();
    let mut t_ej: Vec<f64> = Vec::new();
    let mut max_err: f64 = 0.0;
    for o in &all {
        match o.fate {
            Fate::Ejected(k) => {
                counts[k] += 1;
                t_ej.push(o.t_cr_units);
            }
            Fate::Undecided => counts[3] += 1,
        }
        max_err = max_err.max(o.energy_err);
    }
    steps.sort();
    t_ej.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &[u32], p: f64| v[((v.len() - 1) as f64 * p) as usize];
    println!("cfg {cfg:?}  res {res}  half {half}  time {:?}", start.elapsed());
    println!(
        "fates  A {:.1}%  B {:.1}%  C {:.1}%  undecided {:.1}%",
        100.0 * counts[0] as f64 / n,
        100.0 * counts[1] as f64 / n,
        100.0 * counts[2] as f64 / n,
        100.0 * counts[3] as f64 / n
    );
    println!(
        "steps  p50 {}  p90 {}  p99 {}  max {}  mean {:.0}",
        pct(&steps, 0.5),
        pct(&steps, 0.9),
        pct(&steps, 0.99),
        steps[steps.len() - 1],
        steps.iter().map(|&s| s as f64).sum::<f64>() / n
    );
    if !t_ej.is_empty() {
        println!("median ejection time {:.1} t_cr", t_ej[t_ej.len() / 2]);
    }
    println!("max energy error {max_err:.2e}");
    println!(
        "eta vs eta/2 agreement: early (<10 t_cr) {:.1}% of {early}, later {:.1}% of {late}",
        100.0 * early_ok as f64 / early.max(1) as f64,
        100.0 * late_ok as f64 / late.max(1) as f64
    );
    let mut errs: Vec<f64> = all.iter().map(|o| o.energy_err).collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "energy err p50 {:.1e} p90 {:.1e} p99 {:.1e}",
        errs[errs.len() / 2],
        errs[errs.len() * 9 / 10],
        errs[errs.len() * 99 / 100]
    );
    let capped = all.iter().filter(|o| o.steps >= cfg.max_steps).count();
    let bad = all.iter().filter(|o| o.energy_err > 1e-2).count();
    println!(
        "undecided by step cap {:.1}%  energy err > 1e-2: {:.1}%",
        100.0 * capped as f64 / n,
        100.0 * bad as f64 / n
    );
    // Coarse ASCII map: A = 'a', B = 'b', C = 'c', undecided = '.'
    let stride = (res / 48).max(1);
    for row in rows.iter().step_by(stride) {
        let line: String = row
            .iter()
            .step_by(stride)
            .map(|o| match o.fate {
                Fate::Ejected(0) => 'a',
                Fate::Ejected(1) => 'b',
                Fate::Ejected(_) => 'c',
                Fate::Undecided => '.',
            })
            .collect();
        println!("{line}");
    }
}
