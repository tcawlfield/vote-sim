// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

mod borda;
mod btr_irv;
mod colname;
pub mod condorcet_util;
mod instant_runoff;
mod minimax;
mod multivote;
mod plurality;
mod plurality_top_n;
mod rangevoting;
mod ranked_pairs;
mod results;
mod reweighted_range;
mod star;
mod tallies;
mod test_utils;

pub use borda::Borda;
pub use colname::ColName;
pub use instant_runoff::InstantRunoff;
pub use multivote::Multivote;
pub use plurality::Plurality;
pub use plurality_top_n::PluralityTopN;
pub use rangevoting::RangeVoting;
pub use ranked_pairs::RP;
pub use results::{ElectResult, Standings, Strategy};
pub use reweighted_range::RRV;
pub use star::STAR;

use crate::sim::Sim;
use serde::{Deserialize, Serialize};

/// A single-winner voting method and its parameters, as configured.
///
/// `PartialEq` means "elects the same way": it's how the runner finds a
/// strategic method's honest poll among the configured methods.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Method {
    Plurality(Plurality),
    Range(RangeVoting),
    InstantRunoff(InstantRunoff),
    Borda(Borda),
    Multivote(Multivote),
    STAR(STAR),
    RP(RP),
    BtrIrv(btr_irv::BtrIrv),
    MM(minimax::Minimax),
}

impl Method {
    pub fn new_sim(&self, sim: &Sim) -> Box<dyn MethodSim> {
        match self {
            Method::Plurality(m) => Box::new(m.new_sim(sim)),
            Method::Range(m) => Box::new(m.new_sim(sim)),
            Method::InstantRunoff(m) => Box::new(m.new_sim(sim)),
            Method::Borda(m) => Box::new(m.new_sim(sim)),
            Method::Multivote(m) => Box::new(m.new_sim(sim)),
            Method::STAR(m) => Box::new(m.new_sim(sim)),
            Method::RP(m) => Box::new(m.new_sim(sim)),
            Method::BtrIrv(m) => Box::new(m.new_sim(sim)),
            Method::MM(m) => Box::new(m.new_sim(sim)),
        }
    }

    /// The output column name for this method's results. Also serves as the
    /// method's identity: two configured methods with the same column name
    /// would overwrite each other in the output.
    pub fn colname(&self) -> String {
        match self {
            Method::Plurality(m) => m.colname(),
            Method::Range(m) => m.colname(),
            Method::InstantRunoff(m) => m.colname(),
            Method::Borda(m) => m.colname(),
            Method::Multivote(m) => m.colname(),
            Method::STAR(m) => m.colname(),
            Method::RP(m) => m.colname(),
            Method::BtrIrv(m) => m.colname(),
            Method::MM(m) => m.colname(),
        }
    }

    /// Check the method's parameters; `Config::validate` reports any error
    /// before a simulation starts.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Method::Range(m) => m.validate(),
            Method::Multivote(m) => m.validate(),
            Method::STAR(m) => m.validate(),
            Method::Plurality(_)
            | Method::InstantRunoff(_)
            | Method::Borda(_)
            | Method::RP(_)
            | Method::BtrIrv(_)
            | Method::MM(_) => Ok(()),
        }
    }

    /// The honest method whose result this one takes as its pre-election poll,
    /// for strategic methods whose voters react to the front-runners. `None`
    /// for honest methods and for strategic methods that don't use a poll.
    pub fn honest_poll(&self) -> Option<Method> {
        match self {
            Method::Plurality(m) => m.honest_poll().map(Method::Plurality),
            Method::Range(m) => m.honest_poll().map(Method::Range),
            Method::Borda(m) => m.honest_poll().map(Method::Borda),
            Method::STAR(m) => m.honest_poll().map(Method::STAR),
            Method::InstantRunoff(_)
            | Method::Multivote(_)
            | Method::RP(_)
            | Method::BtrIrv(_)
            | Method::MM(_) => None,
        }
    }
}

pub trait MethodSim {
    /// Run this trial's election, returning every candidate's place. `poll` is
    /// the standings of the method's [`Method::honest_poll`] this trial, and is
    /// `Some` exactly when that is.
    fn elect(&mut self, sim: &Sim, poll: Option<&Standings>) -> &Standings;
    fn name(&self) -> String;
    fn colname(&self) -> String;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum MultiWinMethod {
    RRV(RRV),
    PluralityTopN(PluralityTopN),
}

impl MultiWinMethod {
    pub fn new_sim(&self, sim: &Sim) -> Box<dyn MWMethodSim> {
        match self {
            MultiWinMethod::RRV(m) => Box::new(m.new_sim(sim)),
            MultiWinMethod::PluralityTopN(m) => Box::new(m.new_sim(sim)),
        }
    }

    /// Check the method's parameters; `Config::validate` reports any error
    /// before a simulation starts.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            MultiWinMethod::RRV(m) => m.validate(),
            MultiWinMethod::PluralityTopN(_) => Ok(()),
        }
    }

    /// The output column name for this method's results.
    pub fn colname(&self) -> String {
        match self {
            MultiWinMethod::RRV(m) => m.colname(),
            MultiWinMethod::PluralityTopN(m) => m.colname(),
        }
    }
}

pub trait MWMethodSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult>;
    fn name(&self) -> String;
    fn colname(&self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::considerations::{Consideration, ConsiderationSimKind};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn every_method() -> Vec<Method> {
        [
            r#"{"Plurality": {"strat": "Honest"}}"#,
            r#"{"Plurality": {"strat": "Strategic"}}"#,
            r#"{"Range": {"strat": "Honest", "nranks": 5}}"#,
            r#"{"Range": {"strat": "Strategic", "nranks": 2}}"#,
            r#"{"InstantRunoff": {}}"#,
            r#"{"Borda": {"strat": "Honest"}}"#,
            r#"{"Borda": {"strat": "Strategic"}}"#,
            r#"{"Multivote": {"strat": "Honest", "votes": 3, "spread_fact": 1.0}}"#,
            r#"{"Multivote": {"strat": "Strategic", "votes": 3, "spread_fact": 1.0}}"#,
            r#"{"STAR": {"strat": "Honest"}}"#,
            r#"{"STAR": {"strat": "Strategic"}}"#,
            r#"{"RP": {"strat": "Honest"}}"#,
            r#"{"BtrIrv": {}}"#,
            r#"{"MM": {}}"#,
        ]
        .iter()
        .map(|json| serde_json::from_str(json).unwrap())
        .collect()
    }

    /// Every method's standings place every candidate exactly once, across
    /// field sizes and electorates small enough to tie.
    #[test]
    fn every_method_places_every_candidate_once() {
        let considerations: Vec<Consideration> = serde_json::from_str(
            r#"[{"Likability": {"mean": 0.3}}, {"Issues": [{"sigma": 1.0, "halfcsep": 0.5}]}]"#,
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        for ncand in [2, 3, 6] {
            for nvtr in [2, 5, 51] {
                let mut sim = Sim::new(ncand, nvtr);
                let mut axes: Vec<ConsiderationSimKind> =
                    considerations.iter().map(|c| c.new_sim(&sim)).collect();
                for _ in 0..20 {
                    sim.election(&mut axes, &mut rng);
                    for method in every_method() {
                        let poll = method
                            .honest_poll()
                            .map(|honest| honest.new_sim(&sim).elect(&sim, None).clone());
                        let mut method_sim = method.new_sim(&sim);
                        let standings = method_sim.elect(&sim, poll.as_ref());
                        let mut placed: Vec<usize> =
                            standings.order().iter().map(|r| r.cand).collect();
                        placed.sort_unstable();
                        assert_eq!(
                            placed,
                            (0..ncand).collect::<Vec<_>>(),
                            "{method:?}, {ncand} candidates, {nvtr} voters: {standings:?}"
                        );
                    }
                }
            }
        }
    }
}
