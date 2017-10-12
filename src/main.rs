extern crate rand;
#[macro_use]
extern crate lazy_static;
//#[macro_use(s)]
extern crate ndarray;
use ndarray::{Array2};

mod simvote;

use simvote::*;

const CITIZENS: usize = 5;
const CANDIDATES: usize = 2;

fn main() {
    let mut rng = rand::thread_rng();

    let axes: [&Consideration; 2] = [
        &Likability{
            stretch_factor: 1.0,
        },
        &Issue{
            sigma: 2.0,
        }
    ];
    let mut net_scores: Array2<f64> = Array2::zeros((CITIZENS, CANDIDATES));
    let mut scores = unsafe { Array2::uninitialized((CITIZENS, CANDIDATES)) };
    for ax in axes.iter() {
        ax.gen_scores(&mut scores, &mut rng);
        println!("scores:\n{:?}", scores);
        for (sc, nsc) in scores.iter().zip(net_scores.iter_mut()) {
            *nsc += *sc;
        }
    }
    println!("Net scores:\n{:?}", net_scores);
}
