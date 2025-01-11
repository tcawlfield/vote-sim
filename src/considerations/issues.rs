use super::ConsiderationSim;
use crate::methods::ElectResult;
use crate::sim::Sim;
use ndarray::Array2;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;

const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64; // borrowed from nightly

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Issue {
    pub sigma: f64,
    pub halfcsep: f64,
    pub halfvsep: Option<f64>,
    #[serde(default = "default_false")]
    pub uniform: bool,
}

fn default_false() -> bool {
    false
}

#[derive(Debug)]
pub struct IssuesSim {
    issues: Vec<Issue>,
    cand_position: Array2<f64>,
}

pub fn new_issues_sim(issues: Vec<Issue>, sim: &Sim) -> IssuesSim {
    let num_issues = issues.len();
    IssuesSim {
        issues,
        cand_position: Array2::zeros((sim.ncand, num_issues)),
    }
}

impl ConsiderationSim for IssuesSim {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng, verbose: bool) {
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
        if verbose {
            println!("Candidate positions: {:?}", self.cand_position);
        }
        let mut cit_position = vec![0.0; npos];
        for j in 0..ncit {
            for (ipos, issue) in self.issues.iter().enumerate() {
                cit_position[ipos] = issue.gen_value(&mut rng, true);
            }
            if verbose && ncit < 20 {
                println!("cit {}: {:?}", j, cit_position);
            }
            for i in 0..ncand {
                let mut distsq = 0.0;
                for p in 0..npos {
                    distsq += (self.cand_position[(i, p)] - cit_position[p]).powi(2);
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

    fn push_posn_elements(
        &self,
        report: &mut dyn FnMut(f64, bool),
        final_candidates: Option<&Vec<ElectResult>>,
    ) {
        let (ncand, npos) = self.cand_position.dim();
        if let Some(final_candidates) = final_candidates {
            for fc in final_candidates.iter() {
                for ipos in 0..npos {
                    report(self.cand_position[(fc.cand, ipos)], ipos == npos - 1);
                }
            }
        } else {
            for icand in 0..ncand {
                for ipos in 0..npos {
                    report(self.cand_position[(icand, ipos)], ipos == npos - 1);
                }
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
