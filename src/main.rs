extern crate rand;
#[macro_use]
extern crate lazy_static;

mod simvote;

use simvote::CandLikeability;

fn main() {
    let mut rng = rand::thread_rng();
    let cl = CandLikeability::new(&mut rng);
    println!("cl = {:?}", cl);
}
