use serde::{Deserialize, Serialize};

use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use super::MethodSim;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Minimax {
    // pub strat: Strategy,
}

#[derive(Debug)]
pub struct MinimaxSim {
    _p: Minimax,
    min_victory_margin: Vec<i32>,
    cands: Vec<usize>,
}

impl Minimax {
    pub fn new_sim(&self, sim: &Sim) -> MinimaxSim {
        MinimaxSim {
            _p: self.clone(),
            min_victory_margin: vec![0; sim.ncand],
            cands: (0..sim.ncand).collect(),
        }
    }
}

impl MethodSim for MinimaxSim {
    fn elect(&mut self, sim: &Sim, _honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        // self.min_victory_margin will hold the lowest margin of victory (negative when
        // the candidate loses against another) that the indexing candidate has over others
        // in pair-wise matchups.
        self.min_victory_margin.fill(i32::MAX);
        for (icand, mvm) in self.min_victory_margin.iter_mut().enumerate() {
            for jopnt in 0..sim.ncand {
                if icand != jopnt && sim.i_beats_j_by[(icand, jopnt)] < *mvm {
                    *mvm = sim.i_beats_j_by[(icand, jopnt)];
                }
            }
        }
        // Find winner and runner-up. The winner has the largest worst-margin-of-victory.
        // Candidates in the Smith set will have at least a worst-margin-of-victory of zero,
        // And there will be at least one such candidate.
        let (_, rup, winner) = self.cands.select_nth_unstable_by(sim.ncand - 2, |&a, &b| {
            self.min_victory_margin[a].cmp(&self.min_victory_margin[b])
        });
        WinnerAndRunnerup {
            winner: ElectResult {
                cand: winner[0],
                score: self.min_victory_margin[winner[0]] as f64,
            },
            runnerup: ElectResult {
                cand: *rup,
                score: self.min_victory_margin[*rup] as f64,
            },
        }
    }

    fn name(&self) -> String {
        format!("Minimax, {}", "Honest")
    }

    fn colname(&self) -> String {
        "MM_h".to_string()
    }

    fn strat(&self) -> Strategy {
        Strategy::Honest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::test_utils::sim_from_scores;
    use ndarray::array;

    #[test]
    fn test_minimax() {
        let mut sim = sim_from_scores(&[
            (&[-1., -2., -3., -4.], 42),
            (&[-4., -1., -2., -3.], 26),
            (&[-4., -3., -1., -2.], 15),
            (&[-4., -3., -2., -1.], 17),
        ]);
        sim.rank_candidates();
        assert_eq!(
            sim.i_beats_j_by,
            array![
                [0, -16, -16, -16], // i goes down, j goes across. j > i.
                [16, 0, 36, 36],
                [16, -36, 0, 66],
                [16, -36, -66, 0],
            ]
        );
        let mut method = Minimax {}.new_sim(&sim);
        let honest_results = method.elect(&sim, None);
        assert_eq!(method.min_victory_margin, vec![-16, 16, -36, -66]);
        assert_eq!(honest_results.winner.cand, 1);
        assert_eq!(honest_results.winner.score, 16.);
        assert_eq!(honest_results.runnerup.cand, 0);
        assert_eq!(honest_results.runnerup.score, -16.);
    }
}
