// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use ndarray::Axis;
use serde::{Deserialize, Serialize};

use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use super::MethodSim;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Multivote {
    pub strat: Strategy,
    pub votes: i32,
    pub spread_fact: f64,
}

#[derive(Debug)]
pub struct MultivoteSim {
    p: Multivote,
    tallies: Tallies,
    cand_scores: Vec<f64>,
}

impl Multivote {
    pub fn new_sim(&self, sim: &Sim) -> MultivoteSim {
        MultivoteSim {
            p: self.clone(),
            tallies: vec![0; sim.ncand],
            cand_scores: vec![0.0; sim.ncand],
        }
    }
}

impl MethodSim for MultivoteSim {
    fn elect(&mut self, sim: &Sim, _honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        self.tallies.fill(0);
        if let Strategy::Strategic = self.p.strat {
            for ranks in sim.ranks.axis_iter(Axis(0)) {
                self.tallies[ranks[0]] += self.p.votes;
            }
            return tally_votes(&self.tallies);
        }

        for (icit, utilities) in sim.scores.axis_iter(Axis(0)).enumerate() {
            let min_score = utilities.iter().map(|x| *x).reduce(f64::min).unwrap();
            let ttl_score: f64 = utilities.iter().map(|x| *x).sum();
            self.cand_scores
                .clone_from_slice(utilities.as_slice().unwrap());

            let reduction = self.p.spread_fact * (ttl_score - min_score * sim.ncand as f64)
                / (self.p.votes as f64);
            log::debug!("  voter {} has reduction {}", icit, reduction);
            for _ in 0..self.p.votes {
                let mut max_score = self.cand_scores[0];
                let mut best_cand = 0;
                for i in 1..sim.ncand {
                    if self.cand_scores[i] > max_score {
                        best_cand = i;
                        max_score = self.cand_scores[i];
                    }
                }
                self.tallies[best_cand] += 1;
                self.cand_scores[best_cand] -= reduction;
                log::debug!(
                    "Voter {} votes for {}, scores are now {:?}",
                    icit,
                    best_cand,
                    self.cand_scores
                );
            }
        }

        log::debug!("Multivote tallies are: {:?}", self.tallies);
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        format!("Multivote, {:?}, {} votes", self.p.strat, self.p.votes)
    }

    fn colname(&self) -> String {
        match self.p.strat {
            Strategy::Honest => format!("multi_h_{}v", self.p.votes),
            Strategy::Strategic => format!("multi_s_{}v", self.p.votes),
        }
    }

    fn strat(&self) -> Strategy {
        self.p.strat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    #[test]
    fn test_multivoting() {
        let mut sim = Sim::new(4, 5);
        // min_score is 1, ttl_pts is 14, ttl - min*ncand = 10
        sim.scores = ndarray::array![
            [1.0, 2.0, 4.0, 7.0], // votes 0 1 1 2
            [1.0, 4.0, 2.0, 7.0], //       0 1 1 2
            [7.0, 1.0, 2.0, 4.0], //       2 0 1 1
            [2.0, 1.0, 7.0, 4.0], //       1 0 2 1
            [2.0, 7.0, 1.0, 4.0], //       1 2 0 1
        ];
        // Tallies: 4 4 5 7
        let mut method = Multivote {
            strat: Strategy::Honest,
            votes: 4,
            spread_fact: 1.1,
        }
        .new_sim(&sim);
        // For initial scores of 1., 2., 4.,   7.   -- reduction = 10/4 * 1.1 = 2.75. First vote to 3
        // Next point scores:    1., 2., 4.,   4.25 -- next vote to 3
        // Next scores:          1., 2., 4.,   1.5  -- cand 2 gets the vote
        // Next scores:          1., 2., 1.25, 1.5  -- cand 1 gets the vote
        // Tallies:
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None);
        assert_eq!(honest_results.winner.cand, 3);
        assert_eq!(honest_results.winner.score, 7.);
        assert_eq!(honest_results.runnerup.cand, 2);
        assert_eq!(honest_results.runnerup.score, 5.);

        let mut smethod = Multivote {
            strat: Strategy::Strategic,
            votes: 4,
            spread_fact: 1.1,
        }
        .new_sim(&sim);
        // tallies: 4, 4, 4, 8
        let strat_results = smethod.elect(&sim, Some(honest_results));
        assert_eq!(strat_results.winner.cand, 3);
        assert_eq!(strat_results.winner.score, 8.);
    }
}
