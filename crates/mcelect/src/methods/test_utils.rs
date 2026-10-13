// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use crate::considerations::{Consideration, ConsiderationSimKind};
use crate::sim::Sim;
use ndarray::{ArrayView, s};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[allow(dead_code)]
pub fn sim_from_scores(scores: &[(&[f64], usize)]) -> Sim {
    let ncand = scores[0].0.len();
    let mut nvtr = 0;
    for &vtrs in scores.iter() {
        nvtr += vtrs.1;
    }
    let mut sim = Sim::new(ncand, nvtr);
    let mut ivtr = 0;
    for (vtr_scores, repeat) in scores.iter() {
        let scores_view = ArrayView::from(vtr_scores);
        for _ in 0..*repeat {
            sim.scores.slice_mut(s![ivtr, ..]).assign(&scores_view);
            ivtr += 1;
        }
    }
    sim
}

/// Random elections over a few field sizes, from electorates small enough to
/// tie up to ones that rarely do, each passed to `each`.
#[allow(dead_code)]
pub fn random_elections(mut each: impl FnMut(&Sim)) {
    let considerations: Vec<Consideration> = serde_json::from_str(
        r#"[{"Likability": {"mean": 0.3}}, {"Issues": [{"sigma": 1.0, "halfcsep": 0.5}]}]"#,
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(21);
    for ncand in [2, 3, 6] {
        for nvtr in [2, 5, 51] {
            let mut sim = Sim::new(ncand, nvtr);
            let mut axes: Vec<ConsiderationSimKind> =
                considerations.iter().map(|c| c.new_sim(&sim)).collect();
            for _ in 0..20 {
                sim.election(&mut axes, &mut rng);
                each(&sim);
            }
        }
    }
}
