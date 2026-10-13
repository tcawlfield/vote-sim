// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! A committee method made from any single-winner one: run that election
//! once, and seat the first `k` of its standings.

use serde::{Deserialize, Serialize};

use super::results::ElectResult;
use super::standalone::Standalone;
use super::{MWMethodSim, Method};
use crate::sim::Sim;

/// The first `k` candidates of a single-winner method's standings, as a
/// committee:
///
/// ```toml
/// [[committee_methods]]
/// TopStandings = { RP = { strat = "Honest" } }
/// ```
///
/// Its column is the single-winner method's, prefixed with `top_` (`top_rp_h`
/// here). A strategic method runs its own honest poll, as it would in a
/// single-winner election; its voters still react to the front-runner pair,
/// not to the committee as a whole.
///
/// Ties at the last seat go by candidate order, which in a committee election
/// is an arbitrary label. A tie for first is settled as for a single winner.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct TopStandings(pub Method);

pub struct TopStandingsSim {
    election: Standalone,
    winners: Vec<ElectResult>,
}

impl TopStandings {
    pub fn colname(&self) -> String {
        format!("top_{}", self.0.colname())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate()
    }

    pub fn new_sim(&self, sim: &Sim) -> TopStandingsSim {
        TopStandingsSim {
            election: Standalone::new(&self.0, sim),
            winners: Vec::with_capacity(sim.ncand),
        }
    }
}

impl MWMethodSim for TopStandingsSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult> {
        let standings = self.election.elect(sim);
        self.winners.clear();
        self.winners
            .extend_from_slice(&standings.order()[..nwinners]);
        &self.winners
    }

    fn name(&self) -> String {
        format!("Top standings of {}", self.election.name())
    }

    fn colname(&self) -> String {
        format!("top_{}", self.election.colname())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::test_utils::random_elections;
    use crate::methods::tests::every_method;
    use crate::methods::{MultiWinMethod, PluralityTopN};

    /// The committee is the first `k` of what the method elects on its own:
    /// with its honest poll if strategic, and a tie for first broken the same
    /// way. So a committee of one is the single-winner method's winner.
    #[test]
    fn top_standings_seats_the_methods_first_k() {
        random_elections(|sim| {
            for method in every_method() {
                let poll = method
                    .honest_poll()
                    .map(|honest| honest.new_sim(sim).elect(sim, None).clone());
                let mut expected = method.new_sim(sim).elect(sim, poll.as_ref()).clone();
                sim.break_tie_with_plurality(&mut expected);

                for k in 1..sim.ncand {
                    // A fresh sim, so its coin for ties starts where `expected`'s did.
                    let mut top = TopStandings(method.clone()).new_sim(sim);
                    let seated: Vec<usize> =
                        top.multi_elect(sim, k).iter().map(|w| w.cand).collect();
                    let first_k: Vec<usize> =
                        expected.order()[..k].iter().map(|r| r.cand).collect();
                    assert_eq!(seated, first_k, "{method:?}, k = {k}");
                }
            }
        });
    }

    /// Top standings of honest plurality seat the same committees as
    /// PluralityTopN, the plurality vote's top `k`.
    #[test]
    fn top_plurality_standings_match_plurality_top_n() {
        let plurality: Method =
            serde_json::from_str(r#"{"Plurality": {"strat": "Honest"}}"#).unwrap();
        random_elections(|sim| {
            let mut top = TopStandings(plurality.clone()).new_sim(sim);
            let mut pltn = MultiWinMethod::PluralityTopN(PluralityTopN::default()).new_sim(sim);
            for k in 2..sim.ncand {
                let mut a: Vec<usize> = top.multi_elect(sim, k).iter().map(|w| w.cand).collect();
                let mut b: Vec<usize> = pltn.multi_elect(sim, k).iter().map(|w| w.cand).collect();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "k = {k}");
            }
        });
    }

    #[test]
    fn colname_prefixes_the_methods() {
        let method = |json| TopStandings(serde_json::from_str(json).unwrap());
        assert_eq!(
            method(r#"{"RP": {"strat": "Honest"}}"#).colname(),
            "top_rp_h"
        );
        assert_eq!(
            method(r#"{"STAR": {"strat": "Strategic", "colname": "star_s"}}"#).colname(),
            "top_star_s"
        );
    }
}
