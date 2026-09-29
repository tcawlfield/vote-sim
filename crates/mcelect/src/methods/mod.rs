// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

mod borda;
mod btr_irv;
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
pub use instant_runoff::InstantRunoff;
pub use multivote::Multivote;
pub use plurality::Plurality;
pub use plurality_top_n::PluralityTopN;
pub use rangevoting::RangeVoting;
pub use ranked_pairs::RP;
pub use results::{ElectResult, Strategy, WinnerAndRunnerup};
pub use reweighted_range::RRV;
pub use star::STAR;

use crate::sim::Sim;
use serde::{Deserialize, Serialize};

/// A single-winner voting method and its parameters, as configured.
///
/// `PartialEq` means "elects the same way": it's how the runner finds a
/// strategic method's honest poll among the configured methods.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Run this trial's election. `honest_rslt` is the result of the method's
    /// [`Method::honest_poll`] this trial, and is `Some` exactly when that is.
    fn elect(&mut self, sim: &Sim, honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup;
    fn name(&self) -> String;
    fn colname(&self) -> String;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
