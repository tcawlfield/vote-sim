// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use ndarray::Axis;
use serde::{Deserialize, Serialize};

use super::ColName;
use super::MethodSim;
use super::results::{Strategy, WinnerAndRunnerup, default_honest};
use super::tallies::{Tallies, tally_votes};
use crate::sim::Sim;

/// The Borda Count method is a *ranked* voting method.
/// Every voter ranks each candidate on their ballot from
/// favorite to least favorite.
/// Each ballot contributes a score to the candidates based on
/// the rank-order of that candidate, with the voter's least-favorite
/// getting one point (or zero), the next-least-favorite gets
/// two points (or one), and so on with of course the favorite
/// receiving the most points.
///
/// The method is named after Jean-Charles de Borda who described
/// it in a paper in 1784. He was not the first or last to develop
/// this system.
///
/// This method is not useful for political elections because it is
/// highly susceptable to strategic voting. But it is unusual to
/// study, very simple, and in some informal settings it could
/// be worth considering or at least comparing with.

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Borda {
    /// Strategic or Honest (defaults to Honest).
    /// Strategic ballots rank the top two pre-election (honest) candidates
    /// as first and last depending on the voter's preference between them. The other
    /// candidates are ranked in between these in normal preference order.
    #[serde(default = "default_honest")]
    pub strat: Strategy,
    /// The number of candidates that are ranked on each ballot. This limits
    /// the preference information expressed on the voters' ballots. But it
    /// can be considered for small groups wanting a simplified voting method.
    #[serde(default = "default_none")]
    pub rank_top_n: Option<usize>,
    /// Replaces the default output column name; see [`ColName`].
    #[serde(default, skip_serializing_if = "ColName::is_default")]
    pub colname: ColName,
}

fn default_none() -> Option<usize> {
    None
}

#[derive(Debug)]
pub struct BordaSim {
    p: Borda,
    tallies: Tallies,
}

impl Borda {
    /// The output column name for this method's results: the configured
    /// `colname`, or else the method's default.
    pub fn colname(&self) -> String {
        self.colname.or_else(|| match self.rank_top_n {
            Some(n) => format!("Borda_{}_{}", self.strat.as_letter(), n),
            None => format!("Borda_{}", self.strat.as_letter()),
        })
    }

    /// When strategic, the honest `Borda` whose result serves as this
    /// method's pre-election poll.
    pub fn honest_poll(&self) -> Option<Self> {
        matches!(self.strat, Strategy::Strategic).then(|| Self {
            strat: Strategy::Honest,
            colname: ColName::default(),
            ..self.clone()
        })
    }

    pub fn new_sim(&self, sim: &Sim) -> BordaSim {
        BordaSim {
            p: self.clone(),
            tallies: vec![0; sim.ncand],
        }
    }
}

impl MethodSim for BordaSim {
    fn elect(&mut self, sim: &Sim, honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        self.tallies.fill(0);
        let top_ncand = if let Some(n) = self.p.rank_top_n {
            n
        } else {
            sim.ncand
        };
        match self.p.strat {
            Strategy::Honest => {
                for cand_fav_list in sim.ranks.lanes(Axis(1)) {
                    for (cand_rank, &icand) in cand_fav_list.indexed_iter().take(top_ncand) {
                        self.tallies[icand] += (top_ncand - cand_rank) as i32;
                    }
                }
            }
            Strategy::Strategic => {
                // Note strategic scoring is reduced by 1 so that threat == 0 always.
                let pre_elect = honest_rslt.expect("strategic Borda needs an honest poll");
                for (ivtr, cand_fav_list) in sim.ranks.lanes(Axis(1)).into_iter().enumerate() {
                    let (preferred, threat) = if sim.scores[(ivtr, pre_elect.winner.cand)]
                        >= sim.scores[(ivtr, pre_elect.runnerup.cand)]
                    {
                        (pre_elect.winner.cand, pre_elect.runnerup.cand)
                    } else {
                        (pre_elect.runnerup.cand, pre_elect.winner.cand)
                    };
                    let mut score_shift: i32 = -2; // leave room for preferred to score max
                    for (cand_rank, &icand) in cand_fav_list.indexed_iter() {
                        if icand == preferred {
                            self.tallies[icand] += top_ncand as i32 - 1; // Score preferred the highest
                            score_shift += 1;
                        } else if icand == threat {
                            score_shift += 1;
                        } else if (top_ncand - cand_rank) as i32 + score_shift > 0 {
                            self.tallies[icand] += (top_ncand - cand_rank) as i32 + score_shift;
                        }
                    }
                }
            }
        }
        log::debug!("Borda tallies are: {:?}", self.tallies);
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        match self.p.rank_top_n {
            Some(n) => format!("Borda, {} {}", self.p.strat, n),
            None => format!("Borda, {}", self.p.strat),
        }
    }

    fn colname(&self) -> String {
        self.p.colname()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    #[test]
    fn test_borda_honest_strat() {
        let mut sim = Sim::new(4, 5);
        sim.scores = ndarray::array![
            [4., 3., 2., 1.],
            [1., 4., 2., 3.],
            [3., 4., 2., 1.],
            [3., 2., 1., 4.],
            [3., 2., 4., 1.],
        ];
        // Tallies: 14, 15, 11, 10
        let mut method = Borda {
            strat: Strategy::Honest,
            rank_top_n: None,
            colname: ColName::default(),
        }
        .new_sim(&sim);
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None);
        assert_eq!(honest_results.winner.cand, 1);
        assert_eq!(honest_results.winner.score, 15.);
        assert_eq!(honest_results.runnerup.cand, 0);
        assert_eq!(honest_results.runnerup.score, 14.);

        let mut method = Borda {
            strat: Strategy::Strategic,
            rank_top_n: None,
            colname: ColName::default(),
        }
        .new_sim(&sim);
        let strat_results = method.elect(&sim, Some(honest_results));
        // scores: Note strategic scoring is reduced by 1 so that threat == 0 always
        // 3 0 2 1
        // 0 3 1 2
        // 0 3 2 1
        // 3 0 1 2
        // 3 0 2 1
        assert_eq!(method.tallies, vec![9, 6, 8, 7]);
        assert_eq!(strat_results.winner.cand, 0);
    }

    #[test]
    fn test_borda_ltd() {
        let mut sim = Sim::new(4, 5);
        sim.scores = ndarray::array![
            [4., 3., 2., 1.], // 2 1 0 0
            [1., 4., 2., 3.], // 0 2 0 1
            [3., 4., 2., 1.], // 1 2 0 0
            [3., 2., 1., 4.], // 1 0 0 2
            [2., 3., 4., 1.], // 0 1 2 0
        ];
        // Tallies: 4 6 2 3
        let mut method = Borda {
            strat: Strategy::Honest,
            rank_top_n: Some(2),
            colname: ColName::default(),
        }
        .new_sim(&sim);
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None);
        assert_eq!(method.tallies, vec![4, 6, 2, 3]);
        assert_eq!(honest_results.winner.cand, 1);
        assert_eq!(honest_results.runnerup.cand, 0);

        let mut method = Borda {
            strat: Strategy::Strategic,
            rank_top_n: Some(3),
            colname: ColName::default(),
        }
        .new_sim(&sim);
        let strat_results = method.elect(&sim, Some(honest_results));
        // scores: Note strategic scoring is reduced by 1 so that threat == 0 always
        // 2 0 1 0
        // 0 2 0 1
        // 0 2 1 0
        // 2 0 0 1
        // 0 2 1 0
        assert_eq!(method.tallies, vec![4, 6, 3, 2]);
        assert_eq!(strat_results.winner.cand, 1);
    }
}
