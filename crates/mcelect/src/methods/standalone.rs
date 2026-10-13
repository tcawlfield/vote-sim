// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! Running a single-winner method by itself, for the committee methods built
//! from one.

use super::results::Standings;
use super::{Method, MethodSim};
use crate::sim::Sim;

/// A single-winner method that runs by itself: first its honest poll, if
/// it's strategic, then the method, with a tie for first broken as for any
/// single-winner election. Its standings are then exactly what that method
/// would elect.
pub(super) struct Standalone {
    method: Box<dyn MethodSim>,
    /// The honest poll a strategic method reads, if it has one.
    poll: Option<Box<dyn MethodSim>>,
    /// The latest standings, a tie for first broken.
    standings: Standings,
}

impl Standalone {
    /// Ready to run `method` on fields of `sim`'s size.
    pub(super) fn new(method: &Method, sim: &Sim) -> Standalone {
        Standalone {
            method: method.new_sim(sim),
            poll: method.honest_poll().map(|poll| poll.new_sim(sim)),
            standings: Standings::with_capacity(sim.ncand),
        }
    }

    /// Run the election on `sim`.
    pub(super) fn elect(&mut self, sim: &Sim) -> &Standings {
        let poll = self.poll.as_mut().map(|poll| poll.elect(sim, None));
        self.standings.clone_from(self.method.elect(sim, poll));
        sim.break_tie_with_plurality(&mut self.standings);
        &self.standings
    }

    pub(super) fn name(&self) -> String {
        self.method.name()
    }

    pub(super) fn colname(&self) -> String {
        self.method.colname()
    }
}
