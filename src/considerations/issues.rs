use super::ConsiderationSim;
use crate::sim::Sim;
use ndarray::Array2;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;

const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64; // borrowed from nightly

/// An Issue consideration is an abstract axis of voter preference.
/// Examples may include: conservative versus liberal, authoritarian
/// versus libertarian, ranch dressing versus blue cheese, or hard sci-fi versus
/// soft sci-fi.
///
/// Scores, or voters' perceived utilities for each candidate, are penalized
/// by the distance between the voter and candidate in (Euclidean) issue space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Issue {
    /// The scale of the issue
    pub sigma: f64,
    /// Adds a polarization gap to the candidates. Half are shifted
    /// by -halfcsep, and half by +halfcsep.
    pub halfcsep: f64,
    /// Adds a polarization gap to the citizens. Similar to halfcsep.
    pub halfvsep: Option<f64>,
    /// If uniform=true, citizen and candidate positons are drawn according
    /// to a uniform distribution instead of normal. But the standard
    /// deviation will still equal sigma.
    #[serde(default = "default_false")]
    pub uniform: bool,
    /// If horizon is set, it is the maximum separation along this issue-axis
    /// that a citizen will care about. Any citizen-candidate distance greater than this will
    /// get the same penalty to the perceived utility.
    /// This creates more polarization, as centrists can be disfavored as much
    /// as ideological extremists on the opposite side of a citizen.
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
    cand_position: Array2<f64>,
    horizon_sq: Vec<f64>,
}

pub fn new_issues_sim(issues: Vec<Issue>, sim: &Sim) -> IssuesSim {
    let num_issues = issues.len();
    let horizon_sq = issues.iter().map(|i| i.horizon.powi(2)).collect();
    IssuesSim {
        issues,
        cand_position: Array2::zeros((sim.ncand, num_issues)),
        horizon_sq,
    }
}

impl ConsiderationSim for IssuesSim {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        let npos = self.issues.len();
        for i in 0..ncand {
            for (ipos, issue) in self.issues.iter().enumerate() {
                self.cand_position[(i, ipos)] = issue.gen_value(&mut rng, false);
            }
        }
        log::info!("Candidate positions: {:?}", self.cand_position);
        let mut cit_position = vec![0.0; npos];
        for j in 0..ncit {
            for (ipos, issue) in self.issues.iter().enumerate() {
                cit_position[ipos] = issue.gen_value(&mut rng, true);
            }
            log::info!("cit {}: {:?}", j, cit_position);
            for i in 0..ncand {
                let mut distsq = 0.0;
                for p in 0..npos {
                    let diff = self.cand_position[(i, p)] - cit_position[p];
                    let diffsq = diff * diff;
                    if diffsq < self.horizon_sq[p] {
                        distsq += diffsq;
                    } else {
                        distsq += self.horizon_sq[p];
                    }
                }
                *scores.get_mut((j, i)).unwrap() += -distsq.sqrt();
            }
        }
    }

    fn get_dim(&self) -> usize {
        self.issues.len()
    }

    fn get_name(&self) -> String {
        "issues".to_string()
    }

    fn push_posn_elements(&self, report: &mut dyn FnMut(f64, bool), final_candidates: &Vec<usize>) {
        let (_ncand, npos) = self.cand_position.dim();
        for &fc in final_candidates.iter() {
            for ipos in 0..npos {
                report(self.cand_position[(fc, ipos)], ipos == npos - 1);
            }
        }
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
        if rng.gen::<bool>() {
            sep = -sep;
        }
        if self.uniform {
            let uniform = rand_distr::Uniform::from(-SQRT_3..SQRT_3);
            rng.sample(uniform) * self.sigma + sep
        } else {
            let x: f64 = rng.sample(StandardNormal);
            x * self.sigma + sep
        }
    }
}
