// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use super::ConsiderationSim;
use crate::sim::Sim;
use ndarray::{Array2, azip};
use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64; // borrowed from nightly

/// An Issue consideration is an abstract axis of voter preference.
/// Examples may include: conservative versus liberal, authoritarian
/// versus libertarian, ranch dressing versus blue cheese, or hard sci-fi versus
/// soft sci-fi.
///
/// Scores, or voters' perceived utilities for each choice, are penalized
/// by the distance between the voter and choice in (Euclidean) issue space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Issue {
    /// The scale of the issue
    pub sigma: f64,
    /// A separate sigma for voters if provided
    pub sigma_vtr: Option<f64>,
    /// Adds a polarization gap to the choices. Half are shifted
    /// by -halfcsep, and half by +halfcsep.
    pub halfcsep: f64,
    /// Adds a polarization gap to the voters. Similar to halfcsep.
    pub halfvsep: Option<f64>,
    /// If uniform=true, voter and choice positons are drawn according
    /// to a uniform distribution instead of normal. But the standard
    /// deviation will still equal sigma.
    #[serde(default = "default_false")]
    pub uniform: bool,
    /// If horizon is set, it is the maximum separation along this issue-axis
    /// that a voter will care about. Any voter-choice distance greater than this will
    /// get the same penalty to the perceived utility.
    /// This creates more polarization, as centrists can be disfavored as much
    /// as ideological extremists on the opposite side of a voter.
    #[serde(default = "default_big")]
    pub horizon: f64,
}

fn default_false() -> bool {
    false
}

fn default_big() -> f64 {
    // std::f64::MAX would overflow when squared.
    1.0e100
}

#[derive(Debug)]
pub struct IssuesSim {
    issues: Vec<Issue>,
    choice_positions: Array2<f64>,
    horizon_sq: Vec<f64>,
}

pub fn new_issues_sim(issues: Vec<Issue>, sim: &Sim) -> IssuesSim {
    let num_issues = issues.len();
    let horizon_sq = issues.iter().map(|i| i.horizon.powi(2)).collect();
    IssuesSim {
        issues,
        choice_positions: Array2::zeros((sim.ncand, num_issues)),
        horizon_sq,
    }
}

impl ConsiderationSim for IssuesSim {
    // Called once per election, not in a hot loop. Keeping it out of line stops
    // the generic instantiation from being inlined into the enum dispatch, which
    // otherwise perturbs codegen of the RNG loop below (~12% on the bench).
    #[inline(never)]
    fn add_to_scores<R: Rng + ?Sized>(&mut self, scores: &mut Array2<f64>, mut rng: &mut R) {
        // All voters are the same in this regard.
        // Or at least we assume there are enough voters that every representative
        // group in position-space spans all degrees of likability alignment.
        let npos = self.issues.len();
        for mut choice_row in self.choice_positions.rows_mut() {
            azip!((choice_pos in &mut choice_row, issue in &self.issues) *choice_pos = issue.gen_value(&mut rng, false));
        }
        log::debug!("choice positions: {:?}", self.choice_positions);
        let mut vtr_positions = vec![0.0; npos];
        for vtr_scores in scores.rows_mut() {
            azip!((vtr_pos in &mut vtr_positions, issue in &self.issues) *vtr_pos = issue.gen_value(&mut rng, true));
            self.one_voters_scores(&vtr_positions, vtr_scores);
        }
    }

    fn get_dim(&self) -> usize {
        self.issues.len()
    }

    fn get_name(&self) -> String {
        "issues".to_string()
    }

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_choices: &[usize]) {
        let (_nchoices, npos) = self.choice_positions.dim();
        for &fc in final_choices.iter() {
            for ipos in 0..npos {
                report(self.choice_positions[(fc, ipos)], ipos == npos - 1);
            }
        }
    }
}

impl IssuesSim {
    fn one_voters_scores(
        &mut self,
        vtr_positions: &Vec<f64>,
        mut vtr_scores: ndarray::prelude::ArrayBase<
            ndarray::ViewRepr<&mut f64>,
            ndarray::prelude::Dim<[usize; 1]>,
            f64,
        >,
    ) {
        azip!((choice_posns in self.choice_positions.outer_iter(), ci_ca_score in &mut vtr_scores) {
            azip!((choice_pos in choice_posns, vtr_pos in vtr_positions, hsq in &self.horizon_sq) {
                let diffsq = (choice_pos - vtr_pos) * (choice_pos - vtr_pos);
                if diffsq < *hsq {
                    *ci_ca_score += diffsq;
                } else {
                    *ci_ca_score += *hsq;
                }
            });
            *ci_ca_score = -ci_ca_score.sqrt();
        });
    }
}

impl Issue {
    fn gen_value<R: Rng>(&self, rng: &mut R, is_voter: bool) -> f64 {
        let mut sep = if is_voter {
            match self.halfvsep {
                Some(s) => s,
                None => self.halfcsep,
            }
        } else {
            self.halfcsep
        };
        let sigma = if let Some(vsig) = self.sigma_vtr
            && is_voter
        {
            vsig
        } else {
            self.sigma
        };
        if rng.random::<bool>() {
            sep = -sep;
        }
        if self.uniform {
            rng.random_range(-SQRT_3..=SQRT_3) * sigma + sep
        } else {
            let x: f64 = rng.sample(StandardNormal);
            x * sigma + sep
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Build an IssuesSim directly from choice positions and per-issue
    /// horizons, bypassing the RNG-driven position generation.
    fn make_sim(choice_position: Array2<f64>, horizons: &[f64]) -> IssuesSim {
        let issues = horizons
            .iter()
            .map(|&horizon| Issue {
                sigma: 1.0,
                sigma_vtr: None,
                halfcsep: 0.0,
                halfvsep: None,
                uniform: false,
                horizon,
            })
            .collect();
        IssuesSim {
            issues,
            choice_positions: choice_position,
            horizon_sq: horizons.iter().map(|h| h.powi(2)).collect(),
        }
    }

    #[test]
    fn scores_are_negative_euclidean_distance() {
        // 3 choices in a 2-issue space.
        let mut sim = make_sim(
            array![[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]],
            &[default_big(), default_big()],
        );

        let vtr_positions = vec![0.0, 0.0];
        let mut scores = Array2::zeros((1, 3));
        sim.one_voters_scores(&vtr_positions, scores.row_mut(0));

        assert_eq!(scores.row(0).to_vec(), vec![0.0, -5.0, -1.0]);
    }

    #[test]
    fn scores_accumulate_onto_existing_values() {
        let mut sim = make_sim(array![[0.0], [4.0]], &[default_big()]);

        let vtr_positions = vec![1.0];
        // Pre-seed the row; one_voters_scores adds the squared distances in
        // before taking the (negative) square root.
        let mut scores = array![[3.0, 3.0]];
        sim.one_voters_scores(&vtr_positions, scores.row_mut(0));

        // choice 0: -sqrt(3 + 1) = -2, choice 1: -sqrt(3 + 9) = -sqrt(12)
        assert_eq!(scores[(0, 0)], -2.0);
        assert!((scores[(0, 1)] - -12.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn per_issue_horizon_clamps_the_penalty() {
        // horizon of 2 on each axis => each issue's squared penalty caps at 4.
        let mut sim = make_sim(array![[3.0, 4.0]], &[2.0, 2.0]);

        let vtr_positions = vec![0.0, 0.0];
        let mut scores = Array2::zeros((1, 1));
        sim.one_voters_scores(&vtr_positions, scores.row_mut(0));

        // min(9, 4) + min(16, 4) = 8
        assert!((scores[(0, 0)] - -8.0_f64.sqrt()).abs() < 1e-12);
    }
}
