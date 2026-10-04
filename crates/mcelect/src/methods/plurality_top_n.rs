// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use super::ColName;
use super::MWMethodSim;
use super::tallies::Tallies;
use crate::methods::ElectResult;
use crate::sim::Sim;

/// PluralityTopN is a (bad) multi-winner method based on a plurality
/// ballot. The top N vote-getters are elected.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PluralityTopN {
    /// Replaces the default output column name; see [`ColName`].
    #[serde(default, skip_serializing_if = "ColName::is_default")]
    pub colname: ColName,
}

pub struct PluralityTopNSim {
    _p: PluralityTopN,
    tallies: Tallies,
    winners: Vec<ElectResult>,
    candidates: Vec<usize>,
}

impl PluralityTopN {
    /// The output column name for this method's results: the configured
    /// `colname`, or else the method's default.
    pub fn colname(&self) -> String {
        self.colname.or_else(|| "pltn".to_string())
    }

    pub fn new_sim(&self, sim: &Sim) -> PluralityTopNSim {
        PluralityTopNSim {
            _p: self.clone(),
            tallies: vec![0; sim.ncand],
            winners: Vec::with_capacity(sim.ncand),
            candidates: Vec::with_capacity(sim.ncand),
        }
    }
}

impl MWMethodSim for PluralityTopNSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult> {
        self.tallies.fill(0);
        for &ivtr in sim.ranks.column(0) {
            self.tallies[ivtr] += 1;
        }
        log::debug!("Plurality (top {}) votes: {:?}", nwinners, self.tallies);
        self.candidates.clear();
        self.candidates.extend(0..sim.ncand);
        self.candidates.sort_by_key(|&icand| -self.tallies[icand]);

        self.winners.clear();
        for &icand in self.candidates.iter().take(nwinners) {
            self.winners.push(ElectResult {
                cand: icand,
                score: self.tallies[icand] as f64,
            });
        }
        &self.winners
    }

    fn name(&self) -> String {
        "PluralityTopN".to_string()
    }

    fn colname(&self) -> String {
        self._p.colname()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::ElectResult;
    use crate::sim::Sim;

    #[test]
    fn test_plurality_top_n() {
        // Using a situation described here: https://rangevoting.org/RRVr.html
        let mut sim = Sim::new(4, 6);
        let mut ptn = PluralityTopN::default().new_sim(&sim);
        sim.scores = ndarray::array![
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ];
        sim.rank_candidates();
        // tallies are: 0, 1, 2, 3
        let results = ptn.multi_elect(&sim, 3);
        assert_eq!(results.len(), 3);

        assert_eq!(results[0], ElectResult { cand: 3, score: 3. });

        assert_eq!(results[1], ElectResult { cand: 2, score: 2. },);

        assert_eq!(results[2], ElectResult { cand: 1, score: 1. },);
    }
}
