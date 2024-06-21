use ndarray::Array2;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;
use std::fmt;

pub trait Consideration: fmt::Debug {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, rng: &mut ThreadRng, verbose: bool);
}

//////////////////////////////////////////

#[derive(Debug)]
pub struct Likability {
    pub stretch_factor: f64,
}

// impl Likability {
//     fn gen_cand_like<R: Rng>(&self, rng: &mut R) -> f64 {
//         NORMAL.ind_sample(rng) * self.stretch_factor
//     }
// }

impl Consideration for Likability {
    #[allow(unused_variables)]
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        // let mut candidates = Vec::with_capacity(ncand);
        // for _ in 0..ncand {
        //     candidates.push(self.gen_cand_like(&mut rng));
        // }
        for i in 0..ncand {
            let cand_like: f64 = rng.sample(StandardNormal);
            let cand_like = cand_like * self.stretch_factor;
            for j in 0..ncit {
                //*scores.get_mut((j, i)).unwrap() = *candidates.get(i).unwrap();
                *scores.get_mut((j, i)).unwrap() += cand_like;
            }
        }
    }
}

//////////////////////////////////////////

#[derive(Debug)]
pub struct Issue {
    pub sigma: f64,
    pub halfcsep: f64,
    pub halfvsep: f64,
    cand_position: Vec<f64>,
}

impl Issue {
    pub fn new(sigma: f64, halfcsep: f64, halfvsep: f64, ncand: usize) -> Issue {
        Issue {
            sigma,
            halfcsep,
            halfvsep,
            cand_position: Vec::with_capacity(ncand),
        }
    }
}

fn gen_bimodal_gauss<R: Rng>(sigma: f64, halfsep: f64, rng: &mut R) -> f64 {
    let x: f64 = rng.sample(StandardNormal);
    let x = x * sigma;
    if rng.gen::<bool>() {
        x + halfsep
    } else {
        x - halfsep
    }
}

impl Consideration for Issue {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        for _ in 0..ncand {
            self.cand_position
                .push(gen_bimodal_gauss(self.sigma, self.halfcsep, &mut rng))
        }
        if verbose {
            println!("Candidate positions: {:?}", self.cand_position);
        }
        for j in 0..ncit {
            let cit_position = gen_bimodal_gauss(self.sigma, self.halfvsep, &mut rng);
            for i in 0..ncand {
                *scores.get_mut((j, i)).unwrap() +=
                    -(*self.cand_position.get(i).unwrap() - cit_position).powi(2);
                //-(*self.cand_position.get(i).unwrap() - cit_position).abs(2);
            }
        }
    }
}

//////////////////////////////////////////

#[derive(Debug)]
pub struct MDIssue {
    pub issues: Vec<Issue>,
    cand_position: Array2<f64>,
}

impl MDIssue {
    pub fn new(issues: Vec<Issue>, ncand: usize) -> MDIssue {
        let npos = issues.len();
        MDIssue {
            issues,
            cand_position: Array2::zeros((ncand, npos)),
        }
    }
}

impl Consideration for MDIssue {
    fn add_to_scores(&mut self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        let npos = self.issues.len();
        for i in 0..ncand {
            for p in 0..npos {
                self.cand_position[(i, p)] =
                    gen_bimodal_gauss(self.issues[p].sigma, self.issues[p].halfcsep, &mut rng);
            }
        }
        if verbose {
            println!("Candidate positions: {:?}", self.cand_position);
        }
        let mut cit_position = vec![0.0; npos];
        for j in 0..ncit {
            for p in 0..npos {
                cit_position[p] =
                    gen_bimodal_gauss(self.issues[p].sigma, self.issues[p].halfvsep, &mut rng);
            }
            if verbose && ncit < 20 {
                println!("cit {}: {:?}", j, cit_position);
            }
            for i in 0..ncand {
                let mut distsq = 0.0;
                for p in 0..npos {
                    distsq += (self.cand_position[(i, p)] - cit_position[p]).powi(2);
                }
                scores[(j, i)] = -distsq.sqrt();
            }
        }
    }
}
