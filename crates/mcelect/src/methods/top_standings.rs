// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! A committee method made from any single-winner one: run that election
//! once, and seat the first `k` of its standings.

use serde::{Deserialize, Serialize};

use super::results::{ElectResult, Standings};
use super::{MWMethodSim, Method, MethodSim};
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
    method: Box<dyn MethodSim>,
    /// The honest poll a strategic method reads, if it has one.
    poll: Option<Box<dyn MethodSim>>,
    /// This trial's standings, a tie for first broken.
    standings: Standings,
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
            method: self.0.new_sim(sim),
            poll: self.0.honest_poll().map(|poll| poll.new_sim(sim)),
            standings: Standings::with_capacity(sim.ncand),
            winners: Vec::with_capacity(sim.ncand),
        }
    }
}

impl MWMethodSim for TopStandingsSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult> {
        let poll = self.poll.as_mut().map(|poll| poll.elect(sim, None));
        self.standings.clone_from(self.method.elect(sim, poll));
        sim.break_tie_with_plurality(&mut self.standings);
        self.winners.clear();
        self.winners
            .extend_from_slice(&self.standings.order()[..nwinners]);
        &self.winners
    }

    fn name(&self) -> String {
        format!("Top standings of {}", self.method.name())
    }

    fn colname(&self) -> String {
        format!("top_{}", self.method.colname())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::considerations::{Consideration, ConsiderationSimKind};
    use crate::methods::tests::every_method;
    use crate::methods::{MultiWinMethod, PluralityTopN};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Random elections over a few field sizes, from electorates small enough
    /// to tie up to ones that rarely do.
    fn elections(mut each: impl FnMut(&Sim)) {
        let considerations: Vec<Consideration> = serde_json::from_str(
            r#"[{"Likability": {"mean": 0.3}}, {"Issues": [{"sigma": 1.0, "halfcsep": 0.5}]}]"#,
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(21);
        for ncand in [2, 3, 6] {
            for nvtr in [2, 5, 51] {
                let mut sim = Sim::new(ncand, nvtr);
                let mut axes: Vec<ConsiderationSimKind> =
                    considerations.iter().map(|c| c.new_sim(&sim)).collect();
                for _ in 0..20 {
                    sim.election(&mut axes, &mut rng);
                    each(&sim);
                }
            }
        }
    }

    /// The committee is the first `k` of what the method elects on its own:
    /// with its honest poll if strategic, and a tie for first broken the same
    /// way. So a committee of one is the single-winner method's winner.
    #[test]
    fn top_standings_seats_the_methods_first_k() {
        elections(|sim| {
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
        elections(|sim| {
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
