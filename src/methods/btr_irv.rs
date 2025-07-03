use ndarray::Axis;
use serde::{Deserialize, Serialize};

use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use super::tallies::Tallies;
use super::MethodSim;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BtrIrv {
    // pub strat: Strategy,
}

#[derive(Debug)]
pub struct BtrIrvSim {
    _p: BtrIrv,
    tallies: Tallies,
    eliminated: Vec<bool>,
    candidates: Vec<usize>,
}

impl BtrIrv {
    pub fn new_sim(&self, sim: &Sim) -> BtrIrvSim {
        BtrIrvSim {
            _p: self.clone(),
            tallies: vec![0; sim.ncand],
            eliminated: vec![false; sim.ncand],
            candidates: Vec::with_capacity(sim.ncand),
        }
    }
}

impl MethodSim for BtrIrvSim {
    fn elect(&mut self, sim: &Sim, _honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        self.eliminated.fill(false);
        loop {
            log::debug!("IRV round: eliminated = {:?}", self.eliminated);
            // Tally up the votes -- each voter's favorite non-eliminated candidate gets a tally.
            self.tallies.fill(0);
            for cand_fav_iter in sim.ranks.lanes(Axis(1)) {
                for &icand in cand_fav_iter {
                    if !self.eliminated[icand] {
                        self.tallies[icand] += 1;
                        break;
                    }
                }
            }
            log::debug!("  tallies are: {:?}", self.tallies);

            self.candidates.clear();
            self.candidates.extend(
                self.eliminated
                    .iter()
                    .enumerate()
                    .filter(|(_, &elim)| !elim)
                    .map(|(icand, _)| icand),
            );
            // Sort from low to high tallies
            self.candidates.sort_by_key(|&icand| self.tallies[icand]);
            let top_cand = self.candidates[self.candidates.len() - 1];
            let top_votes = self.tallies[top_cand];
            if self.candidates.len() <= 2 || top_votes >= (sim.ncit as i32 + 1) / 2 {
                let runner_up = self.candidates[self.candidates.len() - 2];
                return WinnerAndRunnerup {
                    winner: ElectResult {
                        cand: top_cand,
                        score: top_votes as f64,
                    },
                    runnerup: ElectResult {
                        cand: runner_up,
                        score: self.tallies[runner_up] as f64,
                    },
                };
            }

            // Eliminate one of the bottom two candidates
            let mut bot_cand = self.candidates[0];
            let bot_cand2 = self.candidates[1];
            if sim.i_beats_j_by[(bot_cand, bot_cand2)] > 0 {
                // Bottom candidate beats the next-lowest in pairwise votes,
                // So we eliminate that one instead. This satisfies both the
                // Condorcet criterion (not eliminating a Condorcet loser) and
                // (possibly?) the Smith criterion (eliminating candidates not in Smith set first).
                bot_cand = bot_cand2;
            }
            log::debug!("top_cand = {}, bot_cand = {}", top_cand, bot_cand);
            self.eliminated[bot_cand] = true;
        }
    }

    fn name(&self) -> String {
        format!("BRT-IRV, {}", "Honest")
    }

    fn colname(&self) -> String {
        "BTR-IRV_h".to_string()
        // match self.p.strat {
        //     Strategy::Honest => "IRV_h".to_string(),
        //     Strategy::Strategic => "IRV_s".to_string(),
        // }
    }

    fn strat(&self) -> Strategy {
        // self.p.strat
        Strategy::Honest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::test_utils::sim_from_scores;

    #[test]
    fn test_btr_irv_honest() {
        let mut sim = sim_from_scores(&[
            (&[-1., -2., -3., -4.], 42),
            (&[-4., -1., -2., -3.], 26),
            (&[-4., -3., -1., -2.], 15),
            (&[-4., -3., -2., -1.], 17),
        ]);
        /* Round 1 tallies: 2, 2, 0, 1 -- eliminate cand 2
         * Round 2 tallies: 2, 2, -, 1 -- eliminate cand 3
         * Round 3 tallies: 3, 2, -, - -- winner is 0, runnerup is 1
         */
        let mut method = BtrIrv {}.new_sim(&sim);
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None);
        assert_eq!(honest_results.winner.cand, 1);
        assert_eq!(honest_results.winner.score, 58.);
        assert_eq!(honest_results.runnerup.cand, 0);
        assert_eq!(honest_results.runnerup.score, 42.);
    }
}
