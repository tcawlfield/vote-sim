mod method_sim;
mod plurality;
mod rangevoting;
mod results;
mod reweighted_range;
mod tallies;

pub use method_sim::MethodSim;
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
}

impl Method {
    pub fn as_sim(&self, sim: &Sim) -> Box<dyn MethodSim> {
        match self {
            Method::Plurality(m) => Box::new(m.new_sim(sim)),
            Method::Range(m) => Box::new(m.new_sim(sim)),
        }
    }
}

pub enum MultiWinMethod {
    RangeVoting,
}
