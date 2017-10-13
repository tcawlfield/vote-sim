extern crate rand;
#[macro_use]
extern crate lazy_static;
//#[macro_use(s)]
extern crate ndarray;
use ndarray::{Array2, Array1};

mod consideration;

use consideration::*;

const CITIZENS: usize = 100000;
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
    //println!("Net scores:\n{:?}", net_scores);
    let plurality_winner = elect_plurality_honest(&net_scores);
    println!("The winner is {}", plurality_winner);
}

fn elect_plurality_honest(net_scores: &Array2<f64>) -> usize {
    //let (ncit, ncand) = net_scores.dim();
    //let mut votes<u32> = Array2::zeros((CANDIDATES));
    let mut votes = [0; CANDIDATES];
    for i in 0..CITIZENS {
        let mut best_cand = 0usize;
        let mut best_score = *net_scores.get((i, 0)).unwrap();
        for j in 1..CANDIDATES {
            let sc = *net_scores.get((i, j)).unwrap();
            if sc > best_score {
                best_score = sc;
                best_cand = j;
            }
        }
        votes[best_cand] += 1;
    }
    println!("votes: {:?}", votes);
    let mut electee = 0usize;
    let mut most_votes = votes[0];
    for j in 1..CANDIDATES {
        if votes[j] > most_votes {
            electee = j;
            most_votes = votes[j];
        }
    }
    electee
}
