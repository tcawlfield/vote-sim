use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

use super::rangevoting::fill_range_ballot;
use super::results::{Strategy, WinnerAndRunnerup};
use super::MWMethodSim;
use crate::methods::ElectResult;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RRV {
    pub strat: Strategy,
    pub ranks: i32,
    pub k: f64,
}

pub struct RRVSim {
    p: RRV,
    wtd_scores: Vec<f64>,
    ballots: Array2<i32>,
    winners: Vec<ElectResult>,
    remaining: Vec<usize>,
}

/*
K = 1.0 favors large political parties. K = 0.5 favors smaller parties (more penalty).
I'm using this purely as a method of spreading out candidates across the position axes.
*/

impl RRV {
    pub fn new_sim(&self, sim: &Sim) -> RRVSim {
        RRVSim {
            p: self.clone(),
            wtd_scores: vec![0.; sim.ncand],
            ballots: Array2::zeros((sim.ncit, sim.ncand)),
            winners: Vec::with_capacity(sim.ncand),
            remaining: Vec::with_capacity(sim.ncand),
        }
    }
}

impl MWMethodSim for RRVSim {
    fn multi_elect(
        &mut self,
        sim: &Sim,
        _honest_rslt: Option<WinnerAndRunnerup>,
        nwinners: usize,
    ) -> &Vec<ElectResult> {
        self.ballots.fill(0);
        for icit in 0..sim.ncit {
            fill_range_ballot(
                &sim.scores.index_axis(Axis(0), icit),
                self.p.ranks,
                self.ballots
                    .index_axis_mut(Axis(0), icit)
                    .as_slice_mut()
                    .unwrap(),
            );
        }

        self.remaining.clear();
        self.remaining.extend(0..sim.ncand);
        self.winners.clear();
        while self.winners.len() < nwinners {
            self.wtd_scores.fill(0.0);
            for i in 0..sim.ncit {
                // Weight is K / (K + SUM/MAX)
                let sum = self
                    .winners
                    .iter()
                    .fold(0, |sum, j| sum + self.ballots[(i, j.cand)]);
                let wt = self.p.k / (self.p.k + (sum as f64) / ((self.p.ranks - 1) as f64));
                for j in self.remaining.iter() {
                    self.wtd_scores[*j] += wt * (self.ballots[(i, *j)] as f64);
                }
            }
            let (winner_idx, winner_score) = {
                let mut rem_iter = self.remaining.iter();
                let mut winner_idx = 0;
                let mut winner_score = self.wtd_scores[*rem_iter.next().unwrap()];
                for (idx, j) in rem_iter.enumerate() {
                    if self.wtd_scores[*j] > winner_score {
                        winner_idx = idx + 1;
                        winner_score = self.wtd_scores[*j];
                    }
                }
                (winner_idx, winner_score)
            };
            let winner = self.remaining.swap_remove(winner_idx);
            self.winners.push(ElectResult {
                cand: winner,
                score: winner_score,
            });
        }
        &self.winners
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;
    use crate::methods::ElectResult;
    use crate::sim::Sim;

    #[test]
    fn test_rrv() {
        // Using a situation described here: https://rangevoting.org/RRVr.html
        let mut sim = Sim::new(5, 100);
        let mut rrv = RRV {
            strat: Strategy::Honest,
            ranks: 11,
            k: 1.0,
        }
        .new_sim(&sim);
        for icit in 0..60 {
            // Team A
            sim.scores[(icit, 0)] = 10.; // A1
            sim.scores[(icit, 1)] = 9.; // A2
            sim.scores[(icit, 2)] = 8.; // A3
            sim.scores[(icit, 3)] = 1.; // B1
            sim.scores[(icit, 4)] = 0.; // B2
        }
        for icit in 60..100 {
            // Team B
            sim.scores[(icit, 0)] = 0.; // A1
            sim.scores[(icit, 1)] = 0.; // A2
            sim.scores[(icit, 2)] = 0.; // A3
            sim.scores[(icit, 3)] = 10.; // B1
            sim.scores[(icit, 4)] = 10.; // B2
        }

        let results = rrv.multi_elect(&sim, None, 3);
        assert_eq!(results.len(), 3);

        // First round, full weights, cand A1 wins with 600 pts (60 * 10 + 40 * 0)
        assert_eq!(
            results[0],
            ElectResult {
                cand: 0,
                score: 600.
            }
        );

        // Round 2
        // Team A got their favorite and are now all deweighted by half: 10 / (10 + 10)
        // Team B got nothing, no deweighting.
        // Cand B1 (idx 3) has 30 downweighted points from Team A, 400 pts from team B
        assert_eq!(
            results[1],
            ElectResult {
                cand: 3,
                score: 430.
            },
        );

        // Round 3
        let a_weight = 10. / (10. + 10. + 1.); // Winner B1 had score of 1 for A group
        let a2_score = a_weight * 9. * 60.;
        // assert_eq!(results[2], ElectResult{cand: 1, score: a2_score});
        assert_eq!(results[2].cand, 1);
        assert_float_eq!(results[2].score, a2_score, ulps <= 2); // forgive last two digits

        // Just checking the score to vote scaling. Would do this sooner but borrow checker whines.
        assert_eq!(rrv.ballots[(0, 0)], 10);
        assert_eq!(rrv.ballots[(0, 4)], 0);
    }
}
