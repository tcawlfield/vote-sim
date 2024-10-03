use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::MethodSim;
use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use crate::sim::Sim;
use super::condorcet_util::{CandPair, find_candidate_pairoffs, lock_in, find_locked_in_winner};

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
        self.pairs.sort_by_key(|p| p.margin);

        self.locked_in.fill(false);
        let winner = self.find_winner(sim, verbose);
        self.pairs.retain(|p| {p.winner == winner || p.loser == winner});
        let runner_up = self.find_winner(sim, false);
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
            Strategy::Honest => "RP_h".to_string(),
            Strategy::Strategic => "RP_s".to_string(),
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}

impl RPSim {
    fn find_winner(&mut self, sim: &Sim, verbose: bool) -> usize {

        let mut pair_iter = self.pairs.iter();

        // Lock in the first two pairs
        lock_in(&mut self.locked_in, pair_iter.next().unwrap(), true);
        lock_in(&mut self.locked_in, pair_iter.next().unwrap(), true);
        if verbose {
            println!("Locked in {:?} and {:?}", self.pairs[0], self.pairs[1]);
        }
        // Lock in remaining pairs provided they do not create a cycle.
        let mut winner = find_locked_in_winner(&mut self.locked_in, sim).unwrap();
        for p in pair_iter {
            lock_in(&mut self.locked_in, p, true);
            match find_locked_in_winner(&mut self.locked_in, sim) {
                Some(w) => {
                    winner = w;
                    if verbose {
                        println!("Locked in {:?} -- current winner is {}", p, w);
                    }
                },
                None => {
                    lock_in(&mut self.locked_in, p, false);
                    if verbose {
                        println!("Won't lock in {:?} -- creates Condorcet cycle", p);
                    }
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
        let mut sim = Sim::new(3, 6);
        sim.scores = ndarray::array![
            [0., 5., 5.],
            [0., 4., 5.],
            [1., 0., 5.],
            [5., 0., 4.],
            [5., 4., 0.],
            [5., 4., 0.],
            // ttls: 16, 17, 19
            // cand 1 prefered to 2: 2 -- 2 vs 1: 3. cand 2 wins the runoff.
        ];
        let mut method = RP {
            strat: Strategy::Honest,
        }
        .new_sim(&sim);
        sim.rank_candidates();
        // cand's 1 and 2 go to runoff.
        let honest_results = method.elect(&sim, None, false);
        assert_eq!(honest_results.winner.cand, 2);
        assert_eq!(honest_results.runnerup.cand, 1);
        assert_eq!(honest_results.winner.score, 3.);
        assert_eq!(honest_results.runnerup.score, 2.);

        sim.scores = ndarray::array![
            [0., 5., 5.],
            [0., 4., 5.],
            [0., 0., 5.],
            [5., 0., 1.],
            [5., 4., 0.],
            [5., 4., 0.],
            // ttls: 15, 17, 16: again 1 and 2 go to runoff but this time we swap.
            // cand 1 prefered to 2: 2 -- 2vs1: 3. cand 2 wins the runoff.
        ];
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None, false);
        assert_eq!(honest_results.winner.cand, 2);
        assert_eq!(honest_results.runnerup.cand, 1);
        assert_eq!(honest_results.winner.score, 3.);
        assert_eq!(honest_results.runnerup.score, 2.);
    }
}
