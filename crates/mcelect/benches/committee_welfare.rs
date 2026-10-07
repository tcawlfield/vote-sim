// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! What committee welfare costs, against the rest of a trial, and whether
//! reusing its buffers across trials matters.
//!
//! For each (candidates, committee size, voters):
//! * `election` -- `Sim::election` with Likability and 2-D Issues: the work a
//!   committee trial already does, as the yardstick.
//! * `prepare_reused` -- `WelfareEval::prepare` (rescale, then search every
//!   committee) with the buffers it allocated up front.
//! * `prepare_fresh` -- the same work with every buffer allocated anew.
//! * `score` -- one method's committee, all welfare functions at once.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mcelect::committee_welfare::{SearchBufs, Welfare, WelfareEval, normalize_into, search};
use mcelect::considerations::{Consideration, ConsiderationSimKind};
use mcelect::sim::Sim;
use ndarray::Array2;
use std::time::Duration;

const WELFARE: [Welfare; 3] = [
    Welfare::Additive,
    Welfare::Harmonic,
    Welfare::ChamberlinCourant,
];

fn considerations(sim: &Sim) -> Vec<ConsiderationSimKind> {
    let configs = [
        r#"{"Likability": {"mean": 0.5}}"#,
        r#"{"Issues": [
            {"sigma": 1.0, "halfcsep": 0.5, "halfvsep": 0.5},
            {"sigma": 0.5, "halfcsep": 0.0}
        ]}"#,
    ];
    configs
        .iter()
        .map(|json| {
            serde_json::from_str::<Consideration>(json)
                .unwrap()
                .new_sim(sim)
        })
        .collect()
}

fn bench_committee_welfare(c: &mut Criterion) {
    let mut group = c.benchmark_group("committee_welfare");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(3));
    for (ncand, k, nvtr) in [(12, 5, 500), (12, 5, 10_000), (16, 5, 500)] {
        let size = format!("{ncand}c{k}k_{nvtr}v");
        let mut sim = Sim::new(ncand, nvtr);
        let mut axes = considerations(&sim);
        let mut rng = rand::rng();
        sim.election(&mut axes, &mut rng);

        group.bench_function(BenchmarkId::new("election", &size), |b| {
            b.iter(|| sim.election(&mut axes, &mut rng))
        });

        let mut eval = WelfareEval::new(&WELFARE, &sim, k);
        group.bench_function(BenchmarkId::new("prepare_reused", &size), |b| {
            b.iter(|| eval.prepare(&sim))
        });

        let weights = Array2::from_shape_fn((WELFARE.len(), k), |(i, j)| WELFARE[i].weights(k)[j]);
        group.bench_function(BenchmarkId::new("prepare_fresh", &size), |b| {
            b.iter(|| {
                let mut norm_t = Array2::zeros((ncand, nvtr));
                let mut bufs = SearchBufs::new(nvtr, k);
                let mut best = vec![0.0; WELFARE.len()];
                let mut mean = vec![0.0; WELFARE.len()];
                normalize_into(sim.scores.view(), &mut norm_t);
                search(
                    norm_t.view(),
                    weights.view(),
                    &mut bufs,
                    &mut best,
                    &mut mean,
                );
                (best, mean)
            })
        });

        let committee: Vec<usize> = (0..k).collect();
        let mut out = vec![0.0; WELFARE.len()];
        group.bench_function(BenchmarkId::new("score", &size), |b| {
            b.iter(|| eval.score(&committee, &mut out))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_committee_welfare);
criterion_main!(benches);
