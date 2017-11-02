use std::fmt;
use rand::distributions::{Normal, IndependentSample};
use rand::{ThreadRng, Rng};
use ndarray::Array2;

lazy_static! {
    static ref NORMAL: Normal = Normal::new(0.0, 1.0);
}

pub trait Consideration: fmt::Debug {
    fn gen_scores(&self, scores: &mut Array2<f64>, &mut ThreadRng, verbose: bool);
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
    fn gen_scores(&self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        // let mut candidates = Vec::with_capacity(ncand);
        // for _ in 0..ncand {
        //     candidates.push(self.gen_cand_like(&mut rng));
        // }
        for i in 0..ncand {
            let cand_like = NORMAL.ind_sample(rng) * self.stretch_factor;
            for j in 0..ncit {
                //*scores.get_mut((j, i)).unwrap() = *candidates.get(i).unwrap();
                *scores.get_mut((j, i)).unwrap() = cand_like;
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
}

fn gen_bimodal_gauss<R: Rng>(sigma: f64, halfsep: f64, rng: &mut R) -> f64 {
    let x = NORMAL.ind_sample(rng) * sigma;
    if rng.gen::<bool>() {
        x + halfsep
    } else {
        x - halfsep
    }
}

impl Consideration for Issue {
    fn gen_scores(&self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        let mut cand_position = Vec::with_capacity(ncand);
        for _ in 0..ncand {
            cand_position.push(gen_bimodal_gauss(self.sigma, self.halfcsep, &mut rng))
        }
        if verbose {
            println!("Candidate positions: {:?}", cand_position);
        }
        for j in 0..ncit {
            let cit_position = gen_bimodal_gauss(self.sigma, self.halfvsep, &mut rng);
            for i in 0..ncand {
                *scores.get_mut((j, i)).unwrap() =
                    -(*cand_position.get(i).unwrap() - cit_position).powi(2);
                    //-(*cand_position.get(i).unwrap() - cit_position).abs(2);
            }
        }
    }
}

//////////////////////////////////////////

#[derive(Debug)]
pub struct MDIssue {
    pub issues: Vec<Issue>,
}

impl Consideration for MDIssue {
    fn gen_scores(&self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng, verbose: bool) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        let npos = self.issues.len();
        let mut cand_position = unsafe { Array2::uninitialized((ncand, npos)) };
        for i in 0..ncand {
            for p in 0..npos {
                cand_position[(i, p)] = gen_bimodal_gauss(self.issues[p].sigma,
                    self.issues[p].halfcsep, &mut rng);
            }
        }
        if verbose {
            println!("Candidate positions: {:?}", cand_position);
        }
        let mut cit_position = vec![0.0; npos];
        for j in 0..ncit {
            for p in 0..npos {
                cit_position[p] = gen_bimodal_gauss(self.issues[p].sigma,
                    self.issues[p].halfvsep, &mut rng);
            }
            if verbose && ncit < 20 {
                println!("cit {}: {:?}", j, cit_position);
            }
            for i in 0..ncand {
                let mut distsq = 0.0;
                for p in 0..npos {
                    distsq += (cand_position[(i,p)] - cit_position[p]).powi(2);
                }
                scores[(j, i)] = -distsq.sqrt();
            }
        }
    }
}

pub fn get_cov_matrix(scores: &Array2<f64>) -> Array2<f64> {
    let (ncit, ncand) = scores.dim();
    let mut mean = vec![0.0; ncand];
    let mut cov_mat: Array2<f64> = Array2::zeros((ncand, ncand));
    for icit in 0..ncit {
        let n = (icit+1) as f64;
        for ix in 0..ncand {
            let dx = scores[(icit, ix)] - mean[ix];
            mean[ix] += dx / n;
            for iy in 0..(ix+1) {
                cov_mat[(ix, iy)] += dx * (scores[(icit, iy)] - mean[iy]);
            }
        }
    }
    for ix in 0..ncand {
        for iy in 0..(ix+1) {
            cov_mat[(ix, iy)] /= (ncit - 1) as f64;
        }
    }

    cov_mat
}
