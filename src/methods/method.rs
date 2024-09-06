use crate::sim::Sim;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectResult {
    pub cand: usize,
    pub score: f64,
}

// WinnerAndRunnerup is useful for strategic methods
#[derive(Debug, Clone, Copy)]
pub struct WinnerAndRunnerup {
    pub winner: ElectResult,
    pub runnerup: ElectResult,
}

#[derive(Debug, Clone, Copy)]
pub enum Strategy {
    Honest,
    Strategic,
}

pub trait Method {
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
