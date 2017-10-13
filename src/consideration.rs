use rand::distributions::{Normal, IndependentSample};
use rand::{ThreadRng, Rng};
use ndarray::Array2;

lazy_static! {
    static ref NORMAL: Normal = Normal::new(0.0, 1.0);
}

pub trait Consideration {
    fn gen_scores(&self, scores: &mut Array2<f64>, &mut ThreadRng);
}

//////////////////////////////////////////

pub struct Likability {
    pub stretch_factor: f64,
}

// impl Likability {
//     fn gen_cand_like<R: Rng>(&self, rng: &mut R) -> f64 {
//         NORMAL.ind_sample(rng) * self.stretch_factor
//     }
// }

impl Consideration for Likability {
    fn gen_scores(&self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng) {
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

pub struct Issue {
    pub sigma: f64,
}

impl Consideration for Issue {
    fn gen_scores(&self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        let mut cand_position = Vec::with_capacity(ncand);
        for _ in 0..ncand {
            cand_position.push(NORMAL.ind_sample(rng) * self.sigma);
        }
        println!("Candidate positions: {:?}", cand_position);
        for j in 0..ncit {
            let cit_position = NORMAL.ind_sample(rng) * self.sigma;
            for i in 0..ncand {
                *scores.get_mut((j, i)).unwrap() =
                    -(*cand_position.get(i).unwrap() - cit_position).powi(2);
                    //-(*cand_position.get(i).unwrap() - cit_position).abs(2);
            }
        }
    }
}
