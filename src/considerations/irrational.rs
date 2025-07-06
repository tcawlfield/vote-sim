use crate::sim::Sim;
use ndarray::Array2;
use rand::distr::StandardUniform;
use rand::rngs::ThreadRng;
use rand::Rng;

use super::ConsiderationSim;

// Irrational is a random utility generator.
// Voters can be assigned to a given number of "camps", where each camp has a
// core set of preferences --
// (All citizens are the same in this regard.)
// Or at least we assume there are enough citizens that every representative
// group in position-space spans all degrees of Irrational alignment.
// If there is a bias in Irrational (Republicans see Trump as highly charismatic)
// then that becomes an issue, not a Irrational.
//
// Irrational is positive and has a mean value of mean.
// Candidate likabilities are <mean> * <standard normal variate>^2

/// A consideration factor that is random for each voter, for each candidate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Irrational {
    /// Scores are uniform distributions, and sigma is the standard deviation.
    /// Uniform variates range from 0 to sqrt(12) * sigma.
    pub sigma: f64,
    /// Voters fall into camps, when camps > 1. (icit % camps) gives the camp index.
    pub camps: usize,
    /// Also when camps > 1, individuals within each camp can deviate from the group.
    /// individualism_deg ranges from 0 to 90, and is an angle in degrees. 0 means
    /// no individualism.
    /// Camp utilities range from 0 to sqrt(12) * sigma * cos(individualism)
    /// Individuals have additional utilities from 0 to sqrt(12) * sigma * sin(individualism)
    pub individualism_deg: f64,
}

#[derive(Debug)]
pub struct IrrationalSim {
    p: Irrational,            // Parameters
    camp_scale: f64,          // cos(individualism)
    individual_scale: f64,    // sin(individualism)
    camp_scores: Array2<f64>, // scores for each (camp, cand)
}

const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64; // borrowed from nightly
const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;
const SQRT12: f64 = 2.0 * SQRT_3;

impl Irrational {
    pub fn new_sim(&self, sim: &Sim) -> IrrationalSim {
        if self.uses_camps() {
            IrrationalSim {
                p: self.clone(),
                camp_scale: f64::cos(self.individualism_deg * RAD_PER_DEG) * self.sigma * SQRT12,
                individual_scale: f64::sin(self.individualism_deg * RAD_PER_DEG)
                    * self.sigma
                    * SQRT12,
                camp_scores: Array2::zeros((self.camps, sim.ncand)),
            }
        } else {
            IrrationalSim {
                p: self.clone(),
                camp_scale: 0.,
                individual_scale: self.sigma * SQRT12,
                camp_scores: Array2::zeros((0, sim.ncand)),
            }
        }
    }

    fn uses_camps(&self) -> bool {
        self.camps > 1
    }
}

impl ConsiderationSim for IrrationalSim {
    #[allow(unused_variables)]
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, rng: &mut ThreadRng) {
        let (ncit, ncand) = scores.dim();
        if self.p.uses_camps() {
            let (ncamps, ncand_from_self) = self.camp_scores.dim();
            assert_eq!(ncand_from_self, ncand);
            for u in self.camp_scores.iter_mut() {
                let uniform_sample: f64 = rng.sample(StandardUniform);
                *u = uniform_sample * self.camp_scale;
            }
            for ((icit, icand), cand_score) in scores.indexed_iter_mut() {
                let icamp = icit % ncamps;
                let uniform_sample: f64 = rng.sample(StandardUniform);
                *cand_score +=
                    self.camp_scores[(icamp, icand)] + uniform_sample * self.individual_scale;
            }
        } else {
            for cand_score in scores.iter_mut() {
                let uniform_sample: f64 = rng.sample(StandardUniform);
                *cand_score += uniform_sample * self.individual_scale;
            }
        }
    }

    fn get_dim(&self) -> usize {
        1
    }

    fn get_name(&self) -> String {
        "Irrational".to_string()
    }

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &Vec<usize>) {
        for _fc in final_candidates.iter() {
            report(f64::NAN, true);
        }
    }
}
