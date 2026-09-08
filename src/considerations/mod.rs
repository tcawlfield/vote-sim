// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use crate::sim::Sim;
use ndarray::Array2;
use rand::Rng;
use std::fmt;

mod irrational;
mod issues;
mod likability;

pub use irrational::{Irrational, IrrationalSim};
pub use issues::{Issue, IssuesSim, new_issues_sim};
pub use likability::{Likability, LikabilitySim};

/// The behavior every consideration's per-election state must provide.
///
/// `add_to_scores` is generic over the RNG rather than taking a concrete
/// `ThreadRng` so that tests can drive it with a seeded, reproducible RNG.
/// The trait is therefore not dyn-safe; static dispatch happens through
/// [`ConsiderationSimKind`].
pub trait ConsiderationSim: fmt::Debug {
    fn add_to_scores<R: Rng + ?Sized>(&mut self, scores: &mut Array2<f64>, rng: &mut R);
    fn get_dim(&self) -> usize;
    fn get_name(&self) -> String;
    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &[usize]);
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Consideration {
    Likability(Likability),
    Issues(Vec<Issue>),
    Irrational(Irrational),
}

impl Consideration {
    pub fn new_sim(&self, sim: &Sim) -> ConsiderationSimKind {
        match self {
            Consideration::Likability(c) => ConsiderationSimKind::Likability(c.new_sim(sim)),
            Consideration::Issues(issues) => {
                ConsiderationSimKind::Issues(new_issues_sim(issues.clone(), sim))
            }
            Consideration::Irrational(c) => ConsiderationSimKind::Irrational(c.new_sim(sim)),
        }
    }
}

/// Statically-dispatched wrapper over every consideration's per-election state.
/// Replaces `Box<dyn ConsiderationSim>` so the hot `add_to_scores` path (millions
/// of RNG draws per trial) has no vtable indirection.
#[derive(Debug)]
pub enum ConsiderationSimKind {
    Likability(LikabilitySim),
    Issues(IssuesSim),
    Irrational(IrrationalSim),
}

macro_rules! dispatch {
    ($self:expr, $inner:ident => $call:expr) => {
        match $self {
            ConsiderationSimKind::Likability($inner) => $call,
            ConsiderationSimKind::Issues($inner) => $call,
            ConsiderationSimKind::Irrational($inner) => $call,
        }
    };
}

impl ConsiderationSim for ConsiderationSimKind {
    fn add_to_scores<R: Rng + ?Sized>(&mut self, scores: &mut Array2<f64>, rng: &mut R) {
        dispatch!(self, c => c.add_to_scores(scores, rng))
    }

    fn get_dim(&self) -> usize {
        dispatch!(self, c => c.get_dim())
    }

    fn get_name(&self) -> String {
        dispatch!(self, c => c.get_name())
    }

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &[usize]) {
        dispatch!(self, c => c.push_posn_elements(report, final_candidates))
    }
}
