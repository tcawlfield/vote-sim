mod borda;
pub mod condorcet_util;
mod instant_runoff;
mod multivote;
mod plurality;
mod rangevoting;
mod ranked_pairs;
mod results;
mod reweighted_range;
mod star;
mod tallies;
mod plurality_top_n;

pub use borda::Borda;
pub use instant_runoff::InstantRunoff;
pub use multivote::Multivote;
pub use plurality::Plurality;
pub use rangevoting::RangeVoting;
pub use ranked_pairs::RP;
pub use results::{ElectResult, Strategy, WinnerAndRunnerup};
pub use reweighted_range::RRV;
pub use star::STAR;
pub use plurality_top_n::PluralityTopN;

use crate::sim::Sim;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Method {
    Plurality(Plurality),
    Range(RangeVoting),
    InstantRunoff(InstantRunoff),
    Borda(Borda),
    Multivote(Multivote),
    STAR(STAR),
    RP(RP),
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
        }
    }
}

pub trait MethodSim {
    fn elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup;
    fn name(&self) -> String;
    fn colname(&self) -> String;
    fn strat(&self) -> Strategy;
}

#[derive(Debug, Serialize, Deserialize)]
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
}

pub trait MWMethodSim {
    fn multi_elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
        nwinners: usize,
        verbose: bool,
    ) -> &Vec<ElectResult>;
}
