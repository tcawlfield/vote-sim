// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use crate::sim::Sim;
use ndarray::Array2;
use rand::distr::StandardUniform;
use rand::{Rng, RngExt as _};

use super::ConsiderationSim;

// Irrational is a random utility generator.
// Voters can be assigned to a given number of "camps", where each camp has a
// core set of preferences --
// (All voters are the same in this regard.)
// Or at least we assume there are enough voters that every representative
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
    /// Voters fall into camps, when camps > 1. (ivtr % camps) gives the camp index.
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
    // See the note on IssuesSim::add_to_scores: out of line on purpose.
    #[inline(never)]
    fn add_to_scores<R: Rng + ?Sized>(&mut self, scores: &mut Array2<f64>, rng: &mut R) {
        let (_nvtr, ncand) = scores.dim();
        if self.p.uses_camps() {
            let (ncamps, ncand_from_self) = self.camp_scores.dim();
            assert_eq!(ncand_from_self, ncand);
            for u in self.camp_scores.iter_mut() {
                let uniform_sample: f64 = rng.sample(StandardUniform);
                *u = uniform_sample * self.camp_scale;
            }
            for ((ivtr, icand), cand_score) in scores.indexed_iter_mut() {
                let icamp = ivtr % ncamps;
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

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &[usize]) {
        for _fc in final_candidates.iter() {
            report(f64::NAN, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn uses_camps_needs_more_than_one_camp() {
        let mut irr = Irrational {
            sigma: 1.0,
            camps: 0,
            individualism_deg: 0.0,
        };
        assert!(!irr.uses_camps());
        irr.camps = 1;
        assert!(!irr.uses_camps());
        irr.camps = 2;
        assert!(irr.uses_camps());
    }

    #[test]
    fn new_sim_without_camps() {
        let sim = Sim::new(4, 10);
        let irr = Irrational {
            sigma: 2.0,
            camps: 1,
            // individualism is meaningless without camps and must be ignored.
            individualism_deg: 45.0,
        };
        let csim = irr.new_sim(&sim);
        assert_eq!(csim.camp_scale, 0.0);
        assert_eq!(csim.individual_scale, 2.0 * SQRT12);
        assert_eq!(csim.camp_scores.dim(), (0, 4));
    }

    #[test]
    fn new_sim_with_camps_splits_scale_by_individualism_angle() {
        let sim = Sim::new(5, 10);
        let irr = Irrational {
            sigma: 3.0,
            camps: 3,
            individualism_deg: 30.0,
        };
        let csim = irr.new_sim(&sim);
        let rad = 30.0 * RAD_PER_DEG;
        assert_abs_diff_eq!(csim.camp_scale, rad.cos() * 3.0 * SQRT12, epsilon = 1e-12);
        assert_abs_diff_eq!(
            csim.individual_scale,
            rad.sin() * 3.0 * SQRT12,
            epsilon = 1e-12
        );
        assert_eq!(csim.camp_scores.dim(), (3, 5));
        // Pythagoras: the two scales combine back to the full sqrt(12) * sigma.
        let combined = csim.camp_scale.hypot(csim.individual_scale);
        assert_abs_diff_eq!(combined, 3.0 * SQRT12, epsilon = 1e-12);
    }

    #[test]
    fn add_to_scores_without_camps_accumulates_within_range() {
        let sim = Sim::new(6, 400);
        let irr = Irrational {
            sigma: 1.0,
            camps: 1,
            individualism_deg: 0.0,
        };
        let mut csim = irr.new_sim(&sim);

        let mut scores = Array2::from_elem((sim.nvtr, sim.ncand), 10.0);
        csim.add_to_scores(&mut scores, &mut rand::rng());

        // Every cell had a uniform sample in [0, sqrt(12) * sigma] added to its
        // starting value of 10.
        for &s in scores.iter() {
            assert!((10.0..=10.0 + SQRT12).contains(&s), "out of range: {s}");
        }
        // With that many samples at least one should land in the top and bottom
        // deciles, proving the values really are spread across the range.
        assert!(scores.iter().any(|&s| s < 10.0 + 0.1 * SQRT12));
        assert!(scores.iter().any(|&s| s > 10.0 + 0.9 * SQRT12));
    }

    #[test]
    fn add_to_scores_without_camps_matches_uniform_moments() {
        let sim = Sim::new(20, 3000);
        let irr = Irrational {
            sigma: 1.0,
            camps: 1,
            individualism_deg: 0.0,
        };
        let mut csim = irr.new_sim(&sim);

        let mut scores = Array2::zeros((sim.nvtr, sim.ncand));
        csim.add_to_scores(&mut scores, &mut rand::rng());

        // Scores are Uniform(0, sqrt(12) * sigma): mean sqrt(3) * sigma, std sigma.
        assert_abs_diff_eq!(scores.mean().unwrap(), SQRT_3, epsilon = 0.1);
        assert_abs_diff_eq!(scores.std(0.0), 1.0, epsilon = 0.1);
    }

    #[test]
    fn add_to_scores_with_camps_is_shared_within_a_camp() {
        let sim = Sim::new(4, 6);
        let irr = Irrational {
            sigma: 1.0,
            camps: 2,
            // No individualism: every voter in a camp gets exactly the camp's scores.
            individualism_deg: 0.0,
        };
        let mut csim = irr.new_sim(&sim);

        let mut scores = Array2::zeros((sim.nvtr, sim.ncand));
        csim.add_to_scores(&mut scores, &mut rand::rng());

        // Voter i belongs to camp (i % 2), so rows 0/2/4 and rows 1/3/5 match.
        for ivtr in 0..sim.nvtr {
            assert_eq!(
                scores.row(ivtr),
                scores.row(ivtr % 2),
                "row {ivtr} should match its camp representative"
            );
        }
        // The two camps drew independently and should differ.
        assert_ne!(scores.row(0), scores.row(1));
        // Camp scores are uniform in [0, sqrt(12) * sigma].
        for &s in scores.iter() {
            assert!((0.0..=SQRT12).contains(&s), "out of range: {s}");
        }
    }

    #[test]
    fn add_to_scores_with_camps_adds_individual_deviation() {
        let sim = Sim::new(4, 6);
        let irr = Irrational {
            sigma: 1.0,
            camps: 2,
            individualism_deg: 90.0, // all individual, no shared camp component
        };
        let mut csim = irr.new_sim(&sim);
        assert_abs_diff_eq!(csim.camp_scale, 0.0, epsilon = 1e-12);

        let mut scores = Array2::zeros((sim.nvtr, sim.ncand));
        csim.add_to_scores(&mut scores, &mut rand::rng());

        // Camp component is zero, so voters in the same camp no longer agree.
        assert_ne!(scores.row(0), scores.row(2));
        for &s in scores.iter() {
            assert!((0.0..=SQRT12).contains(&s), "out of range: {s}");
        }
    }

    #[test]
    fn seeded_rng_makes_add_to_scores_reproducible() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let sim = Sim::new(5, 50);
        let irr = Irrational {
            sigma: 1.5,
            camps: 3,
            individualism_deg: 25.0,
        };
        let run = || {
            let mut csim = irr.new_sim(&sim);
            let mut scores = Array2::zeros((sim.nvtr, sim.ncand));
            csim.add_to_scores(&mut scores, &mut StdRng::seed_from_u64(0xC0FFEE));
            scores
        };
        // Same seed -> bit-for-bit identical scores (the point of the generic RNG).
        assert_eq!(run(), run());
    }

    #[test]
    fn trait_metadata() {
        let sim = Sim::new(3, 4);
        let irr = Irrational {
            sigma: 1.0,
            camps: 1,
            individualism_deg: 0.0,
        };
        let csim = irr.new_sim(&sim);
        assert_eq!(csim.get_dim(), 1);
        assert_eq!(csim.get_name(), "Irrational");

        let mut reported = Vec::new();
        csim.push_posn_elements(&mut |v, last| reported.push((v, last)), &[0, 2]);
        assert_eq!(reported.len(), 2);
        assert!(reported.iter().all(|(v, last)| v.is_nan() && *last));
    }
}
