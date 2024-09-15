use crate::methods::ElectResult;
use crate::sim::Sim;
use ndarray::Array2;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;

use super::ConsiderationSim;

// Likability is an attribute of each candidate that gives them universal appeal.
// (All citizens are the same in this regard.)
// Or at least we assume there are enough citizens that every representative
// group in position-space spans all degrees of likability alignment.
// If there is a bias in likability (Republicans see Trump as highly charismatic)
// then that becomes an issue, not a likability.
//
// Likability is positive and has a mean value of mean.
// Candidate likabilities are <mean> * <standard normal variate>^2

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Likability {
    pub mean: f64,
}

#[derive(Debug)]
pub struct LikabilitySim {
    p: Likability, // Parameters
    scores: Vec<f64>,
}

impl Likability {
    pub fn new_sim(&self, sim: &Sim) -> LikabilitySim {
        LikabilitySim {
            p: self.clone(),
            scores: Vec::with_capacity(sim.ncand),
        }
    }
}

impl ConsiderationSim for LikabilitySim {
    #[allow(unused_variables)]
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();

        self.scores.clear();
        for i in 0..ncand {
            let variant: f64 = rng.sample(StandardNormal);
            let cand_like = variant.powi(2) * self.p.mean;
            self.scores.push(cand_like);
            for j in 0..ncit {
                *scores.get_mut((j, i)).unwrap() += cand_like;
            }
        }
    }

    fn get_dim(&self) -> usize {
        1
    }

    fn get_name(&self) -> String {
        "likability".to_string()
    }

    fn push_posn_elements(
        &self,
        report: &mut dyn FnMut(f64, bool),
        final_candidates: Option<&Vec<ElectResult>>,
    ) {
        if let Some(final_candidates) = final_candidates {
            for fc in final_candidates.iter() {
                report(self.scores[fc.cand], true);
            }
        } else {
            for &score in self.scores.iter() {
                report(score, true);
            }
        }
    }
}
