use rand::distributions::{Normal, IndependentSample};
use rand::Rng;

const LIKEABILITY_DIMS :usize = 3;

lazy_static! {
    static ref NORMAL: Normal = Normal::new(0.0, 1.0);
}

#[derive(Debug)]
pub struct CandLikeability {
    aspect: [f64; LIKEABILITY_DIMS],
}

impl CandLikeability {

    pub fn new<R: Rng>(rng: &mut R) -> CandLikeability {
        let mut cl = CandLikeability{
            aspect: [0.0; LIKEABILITY_DIMS],
        };
        for i in 0..LIKEABILITY_DIMS {
            cl.aspect[i] = NORMAL.ind_sample(rng);
        }
        cl
    }
}
