use ndarray::{ArrayView, Ix1};
use serde::{Deserialize, Serialize};

use super::method_sim::MethodSim;
use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RangeVoting {
    pub strat: Strategy,
    nranks: i32,
}

#[derive(Debug)]
pub struct RangeVotingSim {
    params: RangeVoting,
    tallies: Tallies,
    ballot: Tallies,
}

impl RangeVoting {
    pub fn new_sim(&self, sim: &Sim) -> RangeVotingSim {
        RangeVotingSim {
            params: self.clone(),
            tallies: vec![0; sim.ncand],
            ballot: vec![0; sim.ncand],
        }
    }
}

impl MethodSim for RangeVotingSim {
    fn elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup {
        self.tallies.fill(0);
        // for icit in 0..sim.ncit {
        for vscores in sim.scores.outer_iter() {
            fill_range_ballot(&vscores, self.params.nranks, &mut self.ballot);
            match self.params.strat {
                Strategy::Honest => {
                    for icand in 0..vscores.len() {
                        self.tallies[icand] += self.ballot[icand];
                        // if verbose {
                        //     println!("cand {} has ballot {:?}", icand, self.ballot);
                        // }
                    }
                }
                Strategy::Strategic => {
                    let runnerup = if let Some(pre_election) = honest_rslt {
                        pre_election.runnerup.cand
                    } else {
                        sim.ncand + 9
                    };
                    for icand in 0..vscores.len() {
                        if icand != runnerup {
                            self.tallies[icand] += self.ballot[icand];
                        }
                        // We score the runner-up in the pre-election a zero, not affecting the tallies.
                    }
                }
            }
        }
        if verbose {
            println!("{} tallies: {:?}", self.name(), self.tallies);
        }
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        if self.params.nranks == 2 {
            format!("Approval, {:?}", self.params.strat)
        } else {
            format!("Range 1-{}, {:?}", self.params.nranks, self.params.strat)
        }
    }

    fn colname(&self) -> String {
        if self.params.nranks == 2 {
            match self.params.strat {
                Strategy::Honest => format!("aprv_h"),
                Strategy::Strategic => format!("aprv_s"),
            }
        } else {
            match self.params.strat {
                Strategy::Honest => format!("range_{}_h", self.params.nranks),
                Strategy::Strategic => format!("range_{}_s", self.params.nranks),
            }
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}

pub fn fill_range_ballot(scores: &ArrayView<f64, Ix1>, ranks: i32, ballot: &mut [i32]) {
    let min_score = scores.iter().map(|x| *x).reduce(f64::min).unwrap();
    let max_score = scores.iter().map(|x| *x).reduce(f64::max).unwrap();
    let ranksz = (max_score - min_score) / ((ranks - 1) as f64);
    for (icand, score) in scores.iter().enumerate() {
        // We use a half rank size for scores of 0 and nranks-1.
        let r = ((score - min_score) / ranksz + 0.5).floor() as i32;
        ballot[icand] = r;
    }
}
