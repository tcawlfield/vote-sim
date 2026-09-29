// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::ColName;
use super::MethodSim;
use super::rangevoting::{fill_range_ballot, fill_range_ballot_strat};
use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use super::tallies::{Tallies, tally_votes};
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct STAR {
    pub strat: Strategy,
    #[serde(default = "default_ranks")]
    pub nranks: i32,
    #[serde(default = "default_stretch")]
    strategic_stretch_factor: f64,
    /// Replaces the default output column name; see [`ColName`].
    #[serde(default, skip_serializing_if = "ColName::is_default")]
    pub colname: ColName,
}

fn default_ranks() -> i32 {
    6
}

fn default_stretch() -> f64 {
    4.0
}

#[derive(Debug)]
pub struct STARSim {
    params: STAR,
    tallies: Tallies,
    ballot: Tallies,
    preference_matrix: Array2<i32>,
}

impl STAR {
    /// The output column name for this method's results: the configured
    /// `colname`, or else the method's default.
    pub fn colname(&self) -> String {
        self.colname.or_else(|| match self.strat {
            Strategy::Honest => format!("star_{}_h", self.nranks),
            Strategy::Strategic => format!("star_{}_s", self.nranks),
        })
    }

    /// When strategic, the honest `STAR` whose result serves as this
    /// method's pre-election poll. `strategic_stretch_factor` doesn't affect
    /// honest ballots, so it's reset to its default: the poll then compares
    /// equal to an honest `STAR` configured the ordinary way.
    pub fn honest_poll(&self) -> Option<Self> {
        matches!(self.strat, Strategy::Strategic).then(|| Self {
            strat: Strategy::Honest,
            strategic_stretch_factor: default_stretch(),
            colname: ColName::default(),
            ..self.clone()
        })
    }

    pub fn new_sim(&self, sim: &Sim) -> STARSim {
        STARSim {
            params: self.clone(),
            tallies: vec![0; sim.ncand],
            ballot: vec![0; sim.ncand],
            preference_matrix: Array2::zeros((sim.ncand, sim.ncand)),
        }
    }
}

impl MethodSim for STARSim {
    fn elect(&mut self, sim: &Sim, honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        self.tallies.fill(0);
        self.preference_matrix.fill(0);
        for vscores in sim.scores.outer_iter() {
            match self.params.strat {
                Strategy::Honest => {
                    fill_range_ballot(&vscores, self.params.nranks, &mut self.ballot);
                }
                Strategy::Strategic => {
                    let pre_election = honest_rslt.expect("strategic STAR needs an honest poll");
                    let score_break = (vscores[pre_election.winner.cand]
                        + vscores[pre_election.runnerup.cand])
                        / 2.0;
                    fill_range_ballot_strat(
                        &vscores,
                        self.params.nranks,
                        &mut self.ballot,
                        score_break,
                        self.params.strategic_stretch_factor,
                    );
                }
            }
            for icand in 0..vscores.len() {
                self.tallies[icand] += self.ballot[icand];
                for jcand in icand..sim.ncand {
                    // preference_matrix[(i,j)] counts voters who prefer i to j.
                    // pm[i,j] + pm[j,i] may be less than nvtr because a voter may score i and j equally.
                    if self.ballot[icand] > self.ballot[jcand] {
                        self.preference_matrix[(icand, jcand)] += 1;
                    } else if self.ballot[icand] < self.ballot[jcand] {
                        self.preference_matrix[(jcand, icand)] += 1;
                    }
                }
            }
        }
        log::debug!("{} tallies: {:?}", self.name(), self.tallies);
        let runoff = tally_votes(&self.tallies);
        let ca = runoff.winner.cand;
        let cb = runoff.runnerup.cand;
        if self.preference_matrix[(ca, cb)] >= self.preference_matrix[(cb, ca)] {
            WinnerAndRunnerup {
                winner: ElectResult {
                    cand: runoff.winner.cand,
                    score: self.preference_matrix[(ca, cb)] as f64,
                },
                runnerup: ElectResult {
                    cand: runoff.runnerup.cand,
                    score: self.preference_matrix[(cb, ca)] as f64,
                },
            }
        } else {
            WinnerAndRunnerup {
                winner: ElectResult {
                    cand: runoff.runnerup.cand,
                    score: self.preference_matrix[(cb, ca)] as f64,
                },
                runnerup: ElectResult {
                    cand: runoff.winner.cand,
                    score: self.preference_matrix[(ca, cb)] as f64,
                },
            }
        }
    }

    fn name(&self) -> String {
        format!("STAR 1-{}, {:?}", self.params.nranks, self.params.strat)
    }

    fn colname(&self) -> String {
        self.params.colname()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    #[test]
    fn test_star() {
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
        let mut method = STAR {
            strat: Strategy::Honest,
            nranks: 6,
            strategic_stretch_factor: 1.0,
            colname: ColName::default(),
        }
        .new_sim(&sim);
        // cand's 1 and 2 go to runoff.
        let honest_results = method.elect(&sim, None);
        println!("tallies: {:?}", method.tallies);
        println!("preferences: {:?}", method.preference_matrix);
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
        let honest_results = method.elect(&sim, None);
        println!("tallies: {:?}", method.tallies);
        println!("preferences: {:?}", method.preference_matrix);
        assert_eq!(honest_results.winner.cand, 2);
        assert_eq!(honest_results.runnerup.cand, 1);
        assert_eq!(honest_results.winner.score, 3.);
        assert_eq!(honest_results.runnerup.score, 2.);
    }
}
