extern crate rand;
#[macro_use]
extern crate lazy_static;
//#[macro_use(s)]
extern crate ndarray;
use ndarray::{Array2};

mod consideration;
mod methods;

use consideration::*;
use methods::*;

const CITIZENS: usize = 11;
const CANDIDATES: usize = 5;

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
        //println!("scores:\n{:?}", scores);
        for (sc, nsc) in scores.iter().zip(net_scores.iter_mut()) {
            *nsc += *sc;
        }
    }

    let regs = regrets(&net_scores);

    //println!("Net scores:\n{:?}", net_scores);
    let plh_result = elect_plurality_honest(&net_scores);
    println!("Plurality, honest:");
    print_score(&plh_result, &regs);

    let pls_result = elect_plurality_strategic(&net_scores, 1.0, &plh_result);
    println!("Plurality, strategic:");
    print_score(&pls_result, &regs);

    let r10h_result = elect_range_honest(&net_scores, 10);
    println!("Range<10>, honest:");
    print_score(&r10h_result, &regs);

    let r10s_result = elect_range_strategic(&net_scores, 10, 1.0, &r10h_result);
    println!("Range<10>, strategic:");
    print_score(&r10s_result, &regs);
}

fn print_score(result: &(Result, Result), regs: &Vec<f64>) {
    let r1 = regs[result.0.cand];
    let vic_margin = (result.0.score - result.1.score) / result.0.score;
    println!("  cand {} won, {} is runup, {:.2}% margin, {:.4} regret",
        result.0.cand, result.1.cand, vic_margin * 100.0, r1
    )
}
