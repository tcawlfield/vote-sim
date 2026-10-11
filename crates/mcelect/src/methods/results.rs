// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
// use std::string::ToString;
use strum_macros::Display;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectResult {
    pub cand: usize,
    pub score: f64,
}

/// An election's full result: every candidate in finishing order, winner
/// first, each with the method's score.
///
/// Strategic methods read the front-runners from an honest method's standings
/// (their pre-election poll); a committee could be read off the top.
///
/// The scores are the method's own and mean different things from method to
/// method (votes, points, pairwise margins...). Only the order is common to
/// all. A method sim keeps one, sized for the candidates, and refills it every
/// trial.
#[derive(Debug, PartialEq, Default)]
pub struct Standings {
    order: Vec<ElectResult>,
}

// By hand, so that `clone_from` reuses the buffer: trackers copy their
// method's standings every trial.
impl Clone for Standings {
    fn clone(&self) -> Standings {
        Standings {
            order: self.order.clone(),
        }
    }

    fn clone_from(&mut self, source: &Standings) {
        self.order.clone_from(&source.order);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Strategy {
    Honest,
    Strategic,
}

pub fn default_honest() -> Strategy {
    Strategy::Honest
}

impl Standings {
    pub fn with_capacity(ncand: usize) -> Standings {
        Standings {
            order: Vec::with_capacity(ncand),
        }
    }

    pub fn winner(&self) -> ElectResult {
        self.order[0]
    }

    pub fn runnerup(&self) -> ElectResult {
        self.order[1]
    }

    /// Every candidate, winner first.
    pub fn order(&self) -> &[ElectResult] {
        &self.order
    }

    /// Whether the winner and runner-up have the same score.
    pub fn is_tied(&self) -> bool {
        self.winner().score == self.runnerup().score
    }

    /// Put the runner-up first, for breaking a tie.
    pub fn swap_top_two(&mut self) {
        self.order.swap(0, 1);
    }

    /// Start over, for filling in a new election's result.
    pub(crate) fn clear(&mut self) {
        self.order.clear();
    }

    /// Add the next candidate in finishing order.
    pub(crate) fn push(&mut self, cand: usize, score: f64) {
        self.order.push(ElectResult { cand, score });
    }

    /// The candidates in finishing order, for methods to rearrange in place.
    pub(crate) fn order_mut(&mut self) -> &mut [ElectResult] {
        &mut self.order
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
