// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use super::ColName;
use super::MethodSim;
use super::results::Standings;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Minimax {
    // pub strat: Strategy,
    /// Replaces the default output column name; see [`ColName`].
    #[serde(default, skip_serializing_if = "ColName::is_default")]
    pub colname: ColName,
}

#[derive(Debug)]
pub struct MinimaxSim {
    _p: Minimax,
    min_victory_margin: Vec<i32>,
    cands: Vec<usize>,
    standings: Standings,
}

impl Minimax {
    /// The output column name for this method's results: the configured
    /// `colname`, or else the method's default.
    pub fn colname(&self) -> String {
        self.colname.or_else(|| "MM_h".to_string())
    }

    pub fn new_sim(&self, sim: &Sim) -> MinimaxSim {
        MinimaxSim {
            _p: self.clone(),
            min_victory_margin: vec![0; sim.ncand],
            cands: (0..sim.ncand).collect(),
            standings: Standings::with_capacity(sim.ncand),
        }
    }
}

impl MethodSim for MinimaxSim {
    fn elect(&mut self, sim: &Sim, _honest_rslt: Option<&Standings>) -> &Standings {
        // self.min_victory_margin will hold the lowest margin of victory (negative when
        // the candidate loses against another) that the indexing candidate has over others
        // in pair-wise matchups.
        self.min_victory_margin.fill(i32::MAX);
        for (icand, mvm) in self.min_victory_margin.iter_mut().enumerate() {
            for (jopnt, &margin) in sim.i_beats_j_by.row(icand).iter().enumerate() {
                if icand != jopnt && margin < *mvm {
                    *mvm = margin;
                }
            }
        }
        // Find winner and runner-up. The winner has the largest worst-margin-of-victory.
        // Candidates in the Smith set will have at least a worst-margin-of-victory of zero,
        // And there will be at least one such candidate.
        let mvm = &self.min_victory_margin;
        let (rest, rup, winner) = self
            .cands
            .select_nth_unstable_by(sim.ncand - 2, |&a, &b| mvm[a].cmp(&mvm[b]));
        self.standings.clear();
        self.standings.push(winner[0], mvm[winner[0]] as f64);
        self.standings.push(*rup, mvm[*rup] as f64);
        // The rest by their worst margin, best first. Sorted here rather than
        // in `cands`, whose order carries over to break the next trial's ties.
        for &icand in rest.iter() {
            self.standings.push(icand, mvm[icand] as f64);
        }
        self.standings.order_mut()[2..]
            .sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.cand.cmp(&b.cand)));
        &self.standings
    }

    fn name(&self) -> String {
        format!("Minimax, {}", "Honest")
    }

    fn colname(&self) -> String {
        self._p.colname()
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
        let mut method = Minimax::default().new_sim(&sim);
        let honest_results = method.elect(&sim, None).clone();
        assert_eq!(method.min_victory_margin, vec![-16, 16, -36, -66]);
        assert_eq!(honest_results.winner().cand, 1);
        assert_eq!(honest_results.winner().score, 16.);
        assert_eq!(honest_results.runnerup().cand, 0);
        assert_eq!(honest_results.runnerup().score, -16.);
    }
}
