use super::results::{ElectResult, Strategy, WinnerAndRunnerup};
use crate::sim::Sim;

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

pub trait MWMethodSim {
    fn multi_elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
        nwinners: usize,
        verbose: bool,
    ) -> &Vec<ElectResult>;
}
