extern crate rand;
#[macro_use]
extern crate lazy_static;
//#[macro_use(s)]
extern crate ndarray;
use ndarray::{Array2, Array3};

mod simvote;

use simvote::*;

fn main() {
    let mut rng = rand::thread_rng();

    let axes: [&Consideration; 1] = [
        &Likability{
            stretch_factor: 1.0,
        },
//        Issue{
//            sigma: 2.0,
//        }
    ];
    let mut scores = unsafe { Array2::uninitialized((5, 2)) };
    for ax in axes.iter() {
        ax.gen_scores(&mut scores, &mut rng);
    }
    println!("scores: {:?}", scores);
}
