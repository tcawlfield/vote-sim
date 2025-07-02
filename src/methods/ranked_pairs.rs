use log::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::condorcet_util::{find_candidate_pairoffs, find_locked_in_winner, lock_in, CandPair};
use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use super::MethodSim;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RP {
    pub strat: Strategy,
}

#[derive(Debug)]
pub struct RPSim {
    params: RP,
    pairs: Vec<CandPair>,
    locked_in: Array2<bool>,
}

impl RP {
    pub fn new_sim(&self, sim: &Sim) -> RPSim {
        RPSim {
            params: self.clone(),
            pairs: Vec::with_capacity(sim.ncand * (sim.ncand - 1) / 2),
            locked_in: Array2::default((sim.ncand, sim.ncand)),
        }
    }
}

impl MethodSim for RPSim {
    fn elect(
        &mut self,
        sim: &Sim,
        _honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup {
        find_candidate_pairoffs(&mut self.pairs, sim);
        // Sort by decreasing margin of victory -- first element is the highest-ranked pair.
        self.pairs.sort_by_key(|p| -p.margin);

        let winner = self.find_winner(sim, verbose);
        let runner_up = if sim.ncand > 2 {
            self.pairs
                .retain(|p| p.winner != winner && p.loser != winner);
            self.find_winner(sim, false)
        } else {
            (winner + 1) % 2
        };
        WinnerAndRunnerup {
            winner: ElectResult {
                cand: winner,
                score: 1.0, // don't know a meaningful stat for this.
            },
            runnerup: ElectResult {
                cand: runner_up,
                score: 0.0,
            },
        }
    }

    fn name(&self) -> String {
        format!("RP, {:?}", self.params.strat)
    }

    fn colname(&self) -> String {
        match self.params.strat {
            Strategy::Honest => "rp_h".to_string(),
            Strategy::Strategic => "rp_s".to_string(),
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}

impl RPSim {
    fn find_winner(&mut self, sim: &Sim, verbose: bool) -> usize {
        debug!("Pairs: {:?}", self.pairs);
        let mut pair_iter = self.pairs.iter();
        self.locked_in.fill(false);

        // Lock in the first two pairs
        if let Some(p) = pair_iter.next() {
            lock_in(&mut self.locked_in, p, true);
            if verbose {
                println!("Locked in {:?}", p);
            }
            if let Some(p) = pair_iter.next() {
                lock_in(&mut self.locked_in, p, true);
                if verbose {
                    println!("Locked in {:?}", p);
                }
            }
        }
        // Lock in remaining pairs provided they do not create a cycle.
        let mut winner = find_locked_in_winner(&mut self.locked_in, sim).unwrap();
        for p in pair_iter {
            lock_in(&mut self.locked_in, p, true);
            match find_locked_in_winner(&mut self.locked_in, sim) {
                Some(w) => {
                    winner = w;
                    debug!("Locked in {:?} -- current winner is {}", p, w);
                }
                None => {
                    lock_in(&mut self.locked_in, p, false);
                    debug!("Won't lock in {:?} -- creates Condorcet cycle", p);
                }
            }
        }
        winner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    #[test]
    fn test_rp() {
        // Example from https://electowiki.org/wiki/Ranked_Pairs#Notes
        let mut sim = Sim::new(3, 12);
        sim.scores = ndarray::array![
            [-1., -2., -3.], // 4x A>B>C
            [-1., -2., -3.],
            [-1., -2., -3.],
            [-1., -2., -3.],
            [-3., -1., -2.], // 3x B>C>A
            [-3., -1., -2.],
            [-3., -1., -2.],
            [-2., -3., -1.], // 5x C>A>B
            [-2., -3., -1.],
            [-2., -3., -1.],
            [-2., -3., -1.],
            [-2., -3., -1.],
        ];
        let mut method = RP {
            strat: Strategy::Honest,
        }
        .new_sim(&sim);
        sim.rank_candidates(); // Creates the i_beats_j matrix in sim
        assert_eq!(
            sim.i_beats_j_by,
            ndarray::array![
                [0, 6, -4], // i goes down, j goes across. j > i.
                [-6, 0, 2],
                [4, -2, 0],
            ]
        );

        let honest_results = method.elect(&sim, None, true);
        assert_eq!(honest_results.winner.cand, 2);
    }
}
