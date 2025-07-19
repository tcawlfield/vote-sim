// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
// use std::string::ToString;
use strum_macros::Display;

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Display)]
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

impl Strategy {
    pub fn as_letter(&self) -> &str {
        match self {
            Strategy::Honest => "h",
            Strategy::Strategic => "s",
        }
    }
}
