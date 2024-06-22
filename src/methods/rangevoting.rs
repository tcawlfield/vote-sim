use ndarray::{ArrayView, Ix1};

use super::method::{Method, Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use crate::sim::Sim;

#[derive(Debug)]
pub struct RangeVoting {
    pub strat: Strategy,
    nranks: i32,
    scratch_tallies: Tallies,
    ballot: Tallies,
}

impl RangeVoting {
    pub fn new(sim: &Sim, ranks: i32, strat: Strategy) -> RangeVoting {
        RangeVoting {
            strat,
            nranks: ranks,
            scratch_tallies: vec![0; sim.ncand],
            ballot: vec![0; sim.ncand],
        }
    }
}

impl Method for RangeVoting {
    fn elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup {
        self.scratch_tallies.fill(0);
        // for icit in 0..sim.ncit {
        for vscores in sim.scores.outer_iter() {
            self.fill_ballot(&vscores);
            match self.strat {
                Strategy::Honest => {
                    for icand in 0..vscores.len() {
                        self.scratch_tallies[icand] += self.ballot[icand];
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
                            self.scratch_tallies[icand] += self.ballot[icand];
                        }
                        // We score the runner-up in the pre-election a zero, not affecting the tallies.
                    }
                }
            }
        }
        if verbose {
            println!("{} tallies: {:?}", self.name(), self.scratch_tallies);
        }
        tally_votes(&self.scratch_tallies)
    }

    fn name(&self) -> String {
        if self.nranks == 2 {
            format!("Approval, {:?}", self.strat)
        } else {
            format!("Range 1-{}, {:?}", self.nranks, self.strat)
        }
    }

    fn colname(&self) -> String {
        if self.nranks == 2 {
            match self.strat {
                Strategy::Honest => format!("aprv_h"),
                Strategy::Strategic => format!("aprv_s"),
            }
        } else {
            match self.strat {
                Strategy::Honest => format!("range_{}_h", self.nranks),
                Strategy::Strategic => format!("range_{}_s", self.nranks),
            }
        }
    }

    fn strat(&self) -> Strategy {
        self.strat
    }
}

impl RangeVoting {
    fn fill_ballot(&mut self, scores: &ArrayView<f64, Ix1>) {
        let min_score = scores.iter().map(|x| *x).reduce(f64::min).unwrap();
        let max_score = scores.iter().map(|x| *x).reduce(f64::max).unwrap();
        let ranksz = (max_score - min_score) / ((self.nranks - 1) as f64);
        for (icand, score) in scores.iter().enumerate() {
            // We use a half rank size for scores of 0 and nranks-1.
            let r = ((score - min_score) / ranksz + 0.5).floor() as i32;
            self.ballot[icand] = r;
        }
    }
}
