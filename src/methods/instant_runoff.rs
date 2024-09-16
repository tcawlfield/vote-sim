use ndarray::Axis;
use serde::{Deserialize, Serialize};

use super::method_sim::MethodSim;
use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use super::tallies::Tallies;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstantRunoff {
    // pub strat: Strategy,
}

#[derive(Debug)]
pub struct IRVSim {
    _p: InstantRunoff,
    tallies: Tallies,
    eliminated: Vec<bool>,
}

impl InstantRunoff {
    pub fn new_sim(&self, sim: &Sim) -> IRVSim {
        IRVSim {
            _p: self.clone(),
            tallies: vec![0; sim.ncand],
            eliminated: vec![false; sim.ncand],
        }
    }
}

impl MethodSim for IRVSim {
    fn elect(
        &mut self,
        sim: &Sim,
        _honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup {
        self.eliminated.fill(false);
        loop {
            if verbose {
                println!("IRV round: eliminated = {:?}", self.eliminated);
            }
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
            if verbose {
                println!("  tallies are: {:?}", self.tallies);
            }

            // Find top and bottom candidates
            let mut top_cand = sim.ncand; // invalid index
            let mut bot_cand = sim.ncand;
            let mut top_votes = 0;
            let mut bot_votes = sim.ncit as i32;
            let mut runner_up = sim.ncand;
            let mut runup_votes = 0;
            for (icand, &votes) in self.tallies.iter().enumerate() {
                if self.eliminated[icand] {
                    continue;
                }
                if votes >= top_votes {
                    runner_up = top_cand; // First trip, runner-up still invalid. That's okay.
                    runup_votes = top_votes;
                    top_cand = icand;
                    top_votes = votes;
                    // println!(" - top={}, runup={}", top_cand, runner_up);
                } else if votes > runup_votes {
                    runner_up = icand;
                    runup_votes = votes;
                    // println!(" - runup={}", runner_up);
                }
                if votes < bot_votes {
                    bot_cand = icand;
                    bot_votes = votes;
                    // println!(" - bot_cand={}", bot_cand);
                }
            }

            if verbose {
                println!(
                    "top_cand = {}, runner_up = {}, bot_cand = {}",
                    top_cand, runner_up, bot_cand
                );
            }
            // Do we have an election, or not?
            if top_votes >= (sim.ncit as i32 + 1) / 2 || runner_up == bot_cand {
                return WinnerAndRunnerup {
                    winner: ElectResult {
                        cand: top_cand,
                        score: top_votes as f64,
                    },
                    runnerup: ElectResult {
                        cand: runner_up,
                        score: runup_votes as f64,
                    },
                };
            } else {
                self.eliminated[bot_cand] = true;
            }
        }
    }

    fn name(&self) -> String {
        format!("IRV, {}", "Honest")
    }

    fn colname(&self) -> String {
        "IRV_h".to_string()
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
    use crate::sim::Sim;

    #[test]
    fn test_irv_honest() {
        let mut sim = Sim::new(4, 5);
        sim.scores = ndarray::array![
            [4., 3., 2., 1.],
            [1., 4., 2., 3.],
            [3., 4., 2., 1.],
            [3., 2., 1., 4.],
            [4., 2., 3., 1.],
        ];
        /* Round 1 tallies: 2, 2, 0, 1 -- eliminate cand 2
         * Round 2 tallies: 2, 2, -, 1 -- eliminate cand 3
         * Round 3 tallies: 3, 2, -, - -- winner is 0, runnerup is 1
         */
        let mut method = InstantRunoff {}.new_sim(&sim);
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None, true);
        assert_eq!(honest_results.winner.cand, 0);
        assert_eq!(honest_results.winner.score, 3.);
        assert_eq!(honest_results.runnerup.cand, 1);
        assert_eq!(honest_results.runnerup.score, 2.);
    }
}
