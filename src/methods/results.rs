use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectResult {
    pub cand: usize,
    pub score: f64,
}

// WinnerAndRunnerup is useful for strategic methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WinnerAndRunnerup {
    pub winner: ElectResult,
    pub runnerup: ElectResult,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Strategy {
    Honest,
    Strategic,
}

pub fn default_honest() -> Strategy {
    Strategy::Honest
}

impl WinnerAndRunnerup {
    pub fn is_tied(&self) -> bool {
        self.winner.score == self.runnerup.score
    }
}
