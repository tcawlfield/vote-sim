use crate::sim::Sim;
use ndarray::{s, ArrayView};

pub fn sim_from_scores(scores: &[(&[f64], usize)]) -> Sim {
    let ncand = scores[0].0.len();
    let mut ncit = 0;
    for &cits in scores.iter() {
        ncit += cits.1;
    }
    let mut sim = Sim::new(ncand, ncit);
    let mut icit = 0;
    for (cit_scores, repeat) in scores.iter() {
        let scores_view = ArrayView::from(cit_scores);
        for _ in 0..*repeat {
            sim.scores.slice_mut(s![icit, ..]).assign(&scores_view);
            icit += 1;
        }
    }
    sim
}
