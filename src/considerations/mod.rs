// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use crate::sim::Sim;
use ndarray::Array2;
use rand::Rng;
use std::fmt;

mod electorate;
mod irrational;
mod issues;
mod likability;

pub use electorate::{DistanceFunction, Electorate, ElectorateSim, Faction};
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
    #[serde(alias = "likability")]
    Likability(Likability),
    #[serde(alias = "issues")]
    Issues(Vec<Issue>),
    #[serde(alias = "irrational")]
    Irrational(Irrational),
    #[serde(alias = "electorate")]
    Electorate(Electorate),
}

impl Consideration {
    pub fn new_sim(&self, sim: &Sim) -> ConsiderationSimKind {
        match self {
            Consideration::Likability(c) => ConsiderationSimKind::Likability(c.new_sim(sim)),
            Consideration::Issues(issues) => {
                ConsiderationSimKind::Issues(new_issues_sim(issues.clone(), sim))
            }
            Consideration::Irrational(c) => ConsiderationSimKind::Irrational(c.new_sim(sim)),
            Consideration::Electorate(c) => ConsiderationSimKind::Electorate(c.new_sim(sim)),
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
    Electorate(ElectorateSim),
}

macro_rules! dispatch {
    ($self:expr, $inner:ident => $call:expr) => {
        match $self {
            ConsiderationSimKind::Likability($inner) => $call,
            ConsiderationSimKind::Issues($inner) => $call,
            ConsiderationSimKind::Irrational($inner) => $call,
            ConsiderationSimKind::Electorate($inner) => $call,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn electorate_from_json() {
        let json = r#"
        {
            "electorate": {
                "dimensions": 2,
                "distance_function": {"q_gaussian_2": 1.5},
                "factions": [
                    {
                        "popularity": 0.6,
                        "voter_center": [0.0, 0.0],
                        "voter_spread": 0.5
                    },
                    {
                        "popularity": 0.4,
                        "voter_center": [1.0, 2.0],
                        "voter_spread": 0.5
                    }
                ]
            }
        }
        "#;

        let consid: Consideration = serde_json::from_str(json).unwrap();
        if let Consideration::Electorate(electorate) = consid {
            assert_eq!(electorate.factions.len(), 2);
            assert_eq!(electorate.dimensions, 2);
            assert_eq!(
                electorate.distance_function,
                DistanceFunction::QGaussian2(1.5)
            );
        } else {
            panic!("Expected Factions consideration");
        }
    }

    #[test]
    fn likability_from_json() {
        let json = r#"{ "likability": { "mean": 0.5 } }"#;
        let consid: Consideration = serde_json::from_str(json).unwrap();
        if let Consideration::Likability(likability) = consid {
            assert_eq!(likability.mean, 0.5);
        } else {
            panic!("Expected Likability consideration");
        }
    }

    #[test]
    fn issues_from_json() {
        let json = r#"
        {
            "issues": [
                { "sigma": 1.0, "halfcsep": 0.0 },
                {
                    "sigma": 0.5,
                    "sigma_vtr": 0.7,
                    "halfcsep": 2.0,
                    "halfvsep": 2.0,
                    "uniform": true,
                    "horizon": 3.0
                }
            ]
        }
        "#;

        let consid: Consideration = serde_json::from_str(json).unwrap();
        if let Consideration::Issues(issues) = consid {
            assert_eq!(issues.len(), 2);
            // Terse first entry: optional / defaulted fields.
            assert_eq!(issues[0].sigma, 1.0);
            assert_eq!(issues[0].sigma_vtr, None);
            assert_eq!(issues[0].halfvsep, None);
            assert!(!issues[0].uniform);
            // Fully-specified second entry.
            assert_eq!(issues[1].sigma_vtr, Some(0.7));
            assert!(issues[1].uniform);
            assert_eq!(issues[1].horizon, 3.0);
        } else {
            panic!("Expected Issues consideration");
        }
    }

    #[test]
    fn irrational_from_json() {
        let json = r#"{ "irrational": { "sigma": 1.0, "camps": 3, "individualism_deg": 30.0 } }"#;
        let consid: Consideration = serde_json::from_str(json).unwrap();
        if let Consideration::Irrational(irrational) = consid {
            assert_eq!(irrational.sigma, 1.0);
            assert_eq!(irrational.camps, 3);
            assert_eq!(irrational.individualism_deg, 30.0);
        } else {
            panic!("Expected Irrational consideration");
        }
    }
}
