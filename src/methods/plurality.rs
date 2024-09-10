use super::method_sim::MethodSim;
use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use crate::sim::Sim;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plurality {
    pub strat: Strategy,
}

#[derive(Debug)]
pub struct PluralitySim {
    params: Plurality,
    tallies: Tallies,
}

impl Plurality {
    pub fn new_sim(&self, sim: &Sim) -> PluralitySim {
        PluralitySim {
            params: self.clone(),
            tallies: vec![0; sim.ncand],
        }
    }
}

impl MethodSim for PluralitySim {
    fn elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup {
        match self.params.strat {
            Strategy::Honest => {
                self.tallies.fill(0);
                for icit in 0..sim.ncit {
                    self.tallies[sim.ranks[(icit, 0)]] += 1;
                }
            }
            Strategy::Strategic => {
                let pre_poll = if let Some(prev) = honest_rslt {
                    prev
                } else {
                    self.params.strat = Strategy::Honest;
                    let prev = self.elect(&sim, None, false);
                    self.params.strat = Strategy::Strategic;
                    prev
                };
                self.tallies.fill(0);
                for icit in 0..sim.ncit {
                    for rank in 0..sim.ncand {
                        let icand = sim.ranks[(icit, rank)];
                        if icand == pre_poll.winner.cand || icand == pre_poll.runnerup.cand {
                            self.tallies[icand] += 1;
                            break;
                        }
                    }
                }
            }
        }
        if verbose {
            println!(
                "Plurality votes ({:?}): {:?}",
                self.params.strat, self.tallies
            );
        }
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        format!("Plurality, {:?}", self.params.strat)
    }

    fn colname(&self) -> String {
        match self.params.strat {
            Strategy::Honest => format!("pl_h"),
            Strategy::Strategic => format!("pl_s"),
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}

/*
fn tally_votes_with_plurality_for_ties(
    votes: &Vec<u32>,
    net_scores: &Array2<f64>,
    verbose: bool,
) -> (ElectResult, ElectResult) {
    let (ncit, ncand) = net_scores.dim();
    let mut ntop: usize = 1;
    let mut top_cands: Vec<usize> = vec![0; ncand];
    let mut most_votes = votes[0];

    let mut nsecond: usize = 0;
    let mut second_cands: Vec<usize> = vec![0; ncand];
    let mut runup_votes: u32 = 0;

    for icand in 1..ncand {
        if votes[icand] > most_votes {
            // New best
            nsecond = ntop; // Previous best becomes runner-up
            for i in 0..ntop {
                second_cands[i] = top_cands[i];
            }
            runup_votes = most_votes;

            ntop = 1;
            top_cands[0] = icand;
            most_votes = votes[icand];
        } else if votes[icand] == most_votes {
            // new tie for top
            top_cands[ntop] = icand;
            ntop += 1;
        } else if votes[icand] > runup_votes {
            // new runner-up candidate
            nsecond = 1;
            second_cands[0] = icand;
            runup_votes = votes[icand];
        } else if votes[icand] == runup_votes {
            // new tie for runner-up
            second_cands[nsecond] = icand;
            nsecond += 1;
        }
    }
    if ntop > 1 {
        // Do a runoff
        let mut some_scores = Array2::zeros((ncit, ntop));
        for icit in 0..ncit {
            for icand in 0..ntop {
                some_scores[(icit, icand)] = net_scores[(icit, top_cands[icand])];
            }
        }
        let mut runoff_results = elect_plurality_honest(&some_scores, verbose);
        runoff_results.0.cand = top_cands[runoff_results.0.cand];
        runoff_results.1.cand = top_cands[runoff_results.1.cand];
        runoff_results
    } else if nsecond > 1 {
        // Don't bother to do a runoff.
        let mut rng = rand::thread_rng();
        let chosen_second = rng.gen_range(0..nsecond);
        (
            ElectResult {
                cand: top_cands[0],
                score: most_votes as f64,
            },
            ElectResult {
                cand: second_cands[chosen_second],
                score: runup_votes as f64,
            },
        )
    } else if nsecond == 0 {
        (
            ElectResult {
                cand: top_cands[0],
                score: most_votes as f64,
            },
            ElectResult {
                cand: 0,
                score: -1.0,
            },
        )
    } else {
        (
            ElectResult {
                cand: top_cands[0],
                score: most_votes as f64,
            },
            ElectResult {
                cand: second_cands[0],
                score: runup_votes as f64,
            },
        )
    }
}
*/
