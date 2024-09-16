mod instant_runoff;
mod method_sim;
mod plurality;
mod rangevoting;
mod results;
mod reweighted_range;
mod tallies;

pub use instant_runoff::InstantRunoff;
pub use method_sim::{MWMethodSim, MethodSim};
pub use plurality::Plurality;
pub use rangevoting::RangeVoting;
pub use results::{ElectResult, Strategy, WinnerAndRunnerup};
pub use reweighted_range::RRV;

use crate::sim::Sim;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Method {
    Plurality(Plurality),
    Range(RangeVoting),
    InstantRunoff(InstantRunoff),
}

impl Method {
    pub fn new_sim(&self, sim: &Sim) -> Box<dyn MethodSim> {
        match self {
            Method::Plurality(m) => Box::new(m.new_sim(sim)),
            Method::Range(m) => Box::new(m.new_sim(sim)),
            Method::InstantRunoff(m) => Box::new(m.new_sim(sim)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MultiWinMethod {
    RRV(RRV),
}

impl MultiWinMethod {
    pub fn new_sim(&self, sim: &Sim) -> Box<dyn MWMethodSim> {
        match self {
            MultiWinMethod::RRV(m) => Box::new(m.new_sim(sim)),
        }
    }
}
