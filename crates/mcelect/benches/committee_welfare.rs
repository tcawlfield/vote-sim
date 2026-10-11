// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! What committee welfare costs, against the rest of a trial.
//!
//! For each (candidates, committee size, voters):
//! * `election` -- `Sim::election` with Likability and 2-D Issues: the work a
//!   committee trial already does, as the yardstick.
//! * `prepare_all` -- `WelfareEval::prepare` for Additive, Harmonic and
//!   Chamberlin-Courant: rescale, find each mean, and find each best -- by a
//!   top-k for Additive and branch and bound for the others.
//! * `prepare_exhaustive` -- the same for an `Owa` with rising weights, whose
//!   best is found by visiting every committee: the yardstick for the others.
//! * `score` -- one method's committee, all welfare functions at once.
//!
//! How much branch and bound prunes varies from election to election, so the
//! `prepare` benchmarks cycle through a batch of them.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mcelect::committee_welfare::{Welfare, WelfareEval};
use mcelect::considerations::{Consideration, ConsiderationSimKind};
use mcelect::sim::Sim;
use std::time::Duration;

/// Elections the `prepare` benchmarks cycle through.
const ELECTIONS: usize = 16;

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

        let elections: Vec<Sim> = (0..ELECTIONS)
            .map(|_| {
                let mut sim = Sim::new(ncand, nvtr);
                sim.election(&mut axes, &mut rng);
                sim
            })
            .collect();
        let rising = [Welfare::Owa {
            weights: vec![0.1, 0.2, 1.0],
            colname: "rising".to_string(),
        }];
        for (name, welfare) in [
            ("prepare_all", &WELFARE[..]),
            ("prepare_exhaustive", &rising[..]),
        ] {
            let mut eval = WelfareEval::new(welfare, &sim, k);
            let mut next = elections.iter().cycle();
            group.bench_function(BenchmarkId::new(name, &size), |b| {
                b.iter(|| eval.prepare(next.next().unwrap()))
            });
        }

        let mut eval = WelfareEval::new(&WELFARE, &sim, k);
        eval.prepare(&sim);

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
