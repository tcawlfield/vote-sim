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

impl Likability {
    fn gen_cand_like<R: Rng>(&self, rng: &mut R) -> f64 {
        NORMAL.ind_sample(rng) * self.stretch_factor
    }
}

impl Consideration for Likability {
    fn gen_scores(&self, scores: &mut Array2<f64>, mut rng: &mut ThreadRng) {
        let (ncit, ncand) = scores.dim();
        // All citizens are the same in this regard.
        // Or at least we assume there are enough citizens that every representative
        // group in position-space spans all degrees of likability alignment.
        let mut candidates = Vec::with_capacity(ncand);
        for i in 0..ncand {
            candidates.push(self.gen_cand_like(&mut rng));
        }
        for j in 0..ncit {
            for i in 0..ncand {
                *scores.get_mut((j, i)).unwrap() = *candidates.get(i).unwrap();
            }
        }
    }
}

//////////////////////////////////////////
