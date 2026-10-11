// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::ColName;
use super::MethodSim;
use super::rangevoting::{fill_range_ballot, fill_range_ballot_strat};
use super::results::{Standings, Strategy};
use super::tallies::{Tallies, tally_votes};
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
    standings: Standings,
    ballot: Tallies,
    preference_matrix: Array2<i32>,
}

impl STAR {
    /// Check the parameters: like Range, a ballot needs at least two scores.
    pub fn validate(&self) -> Result<(), String> {
        if self.nranks < 2 {
            return Err(format!("STAR needs nranks >= 2, not {}", self.nranks));
        }
        Ok(())
    }

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
            standings: Standings::with_capacity(sim.ncand),
            ballot: vec![0; sim.ncand],
            preference_matrix: Array2::zeros((sim.ncand, sim.ncand)),
        }
    }
}

impl MethodSim for STARSim {
    fn elect(&mut self, sim: &Sim, honest_rslt: Option<&Standings>) -> &Standings {
        self.tallies.fill(0);
        self.preference_matrix.fill(0);
        for vscores in sim.scores.outer_iter() {
            match self.params.strat {
                Strategy::Honest => {
                    fill_range_ballot(&vscores, self.params.nranks, &mut self.ballot);
                }
                Strategy::Strategic => {
                    let pre_election = honest_rslt.expect("strategic STAR needs an honest poll");
                    let score_break = (vscores[pre_election.winner().cand]
                        + vscores[pre_election.runnerup().cand])
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
        // The top two by score go to a runoff; everyone else keeps their place
        // by score, and their score total.
        tally_votes(&self.tallies, &mut self.standings);
        let ca = self.standings.winner().cand;
        let cb = self.standings.runnerup().cand;
        let (ca_prefs, cb_prefs) = (
            self.preference_matrix[(ca, cb)],
            self.preference_matrix[(cb, ca)],
        );
        let top_two = &mut self.standings.order_mut()[..2];
        top_two[0].score = ca_prefs as f64;
        top_two[1].score = cb_prefs as f64;
        if ca_prefs < cb_prefs {
            self.standings.swap_top_two();
        }
        &self.standings
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
        let honest_results = method.elect(&sim, None).clone();
        println!("tallies: {:?}", method.tallies);
        println!("preferences: {:?}", method.preference_matrix);
        assert_eq!(honest_results.winner().cand, 2);
        assert_eq!(honest_results.runnerup().cand, 1);
        assert_eq!(honest_results.winner().score, 3.);
        assert_eq!(honest_results.runnerup().score, 2.);

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
        let honest_results = method.elect(&sim, None).clone();
        println!("tallies: {:?}", method.tallies);
        println!("preferences: {:?}", method.preference_matrix);
        assert_eq!(honest_results.winner().cand, 2);
        assert_eq!(honest_results.runnerup().cand, 1);
        assert_eq!(honest_results.winner().score, 3.);
        assert_eq!(honest_results.runnerup().score, 2.);
    }

    /// The runoff pair comes first, by its head-to-head votes; everyone else
    /// follows by score total.
    #[test]
    fn star_places_the_rest_by_score() {
        let mut sim = Sim::new(4, 3);
        // Scored 0, 5, 3, 4 on a 0-5 ballot by all three voters.
        sim.scores = ndarray::Array2::from_shape_fn((3, 4), |(_, c)| [0.0, 1.0, 0.5, 0.75][c]);
        let mut method = STAR {
            strat: Strategy::Honest,
            nranks: 6,
            strategic_stretch_factor: 1.0,
            colname: ColName::default(),
        }
        .new_sim(&sim);
        let standings = method.elect(&sim, None);
        let places: Vec<(usize, f64)> = standings
            .order()
            .iter()
            .map(|r| (r.cand, r.score))
            .collect();
        // 1 beats 3 in the runoff 3 voters to 0; then 2 (9 points) and 0 (none).
        assert_eq!(places, [(1, 3.0), (3, 0.0), (2, 9.0), (0, 0.0)]);
    }
}
