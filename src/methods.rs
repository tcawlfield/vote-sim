// Voting methods
use ndarray::{Array2, ArrayView, Axis, Ix1};
use std::f64;

#[derive(Debug)]
pub struct Result {
    pub cand: usize,
    pub score: f64,
}

pub fn elect_plurality_honest(net_scores: &Array2<f64>) -> (Result, Result) {
    let (ncit, ncand) = net_scores.dim();
    //let mut votes<u32> = Array2::zeros((CANDIDATES));
    //let mut votes = [0; CANDIDATES];
    let mut votes = vec![0u32; ncand];
    for i in 0..ncit {
        let mut best_cand = 0usize;
        let mut best_score = *net_scores.get((i, 0)).unwrap();
        for j in 1..ncand {
            let sc = *net_scores.get((i, j)).unwrap();
            if sc > best_score {
                best_score = sc;
                best_cand = j;
            }
        }
        votes[best_cand] += 1;
    }
    println!("honest plurality votes: {:?}", votes);
    tally_votes(&votes)
}

fn tally_votes(votes: &Vec<u32>) -> (Result, Result) {
    let ncand = votes.len();
    let mut electee = 0usize;
    let mut most_votes = votes[0];
    let mut runup = 1usize;
    let mut runup_votes = votes[1];
    if runup_votes > most_votes {
        electee = 1;
        runup = 0;
        most_votes = votes[1];
        runup_votes = votes[0];
    }
    for j in 2..ncand {
        if votes[j] > most_votes {
            runup = electee;
            runup_votes = most_votes;
            electee = j;
            most_votes = votes[j];
        } else if votes[j] > runup_votes {
            runup = j;
            runup_votes = votes[j];
        }
    }
    (Result{cand: electee, score: most_votes as f64}, Result{cand: runup, score: runup_votes as f64})
}

pub fn elect_plurality_strategic(net_scores: &Array2<f64>, frac_strategic: f64,
        pre_results: &(Result, Result)) -> (Result, Result) {
    let (ncit, ncand) = net_scores.dim();
    //let mut votes<u32> = Array2::zeros((CANDIDATES));
    //let mut votes = [0; CANDIDATES];
    let mut votes = vec![0u32; ncand];
    let last_strategic = (ncit as f64 * frac_strategic).round() as usize;
    for i in 0..ncit {
        if i < last_strategic {
            // strategic voter. Which of the top two candidates do you like best?
            let cand1 = pre_results.0.cand;
            let cand2 = pre_results.1.cand;
            if net_scores[(i, cand1)] >= net_scores[(i, cand2)] {
                votes[cand1] += 1;
            } else {
                votes[cand2] += 1;
            }
        } else {
            // honest voter
            let mut best_cand = 0usize;
            let mut best_score = *net_scores.get((i, 0)).unwrap();
            for j in 1..ncand {
                let sc = *net_scores.get((i, j)).unwrap();
                if sc > best_score {
                    best_score = sc;
                    best_cand = j;
                }
            }
            votes[best_cand] += 1;
        }
    }
    println!("strategic plurality votes: {:?}", votes);
    tally_votes(&votes)
}

pub fn regrets(net_scores: &Array2<f64>) -> Vec<f64> {
    let (ncit, ncand) = net_scores.dim();
    let mut utilities = Vec::with_capacity(ncand);
    let mut max_util = f64::MIN;
    let mut avg_util = 0.0;
    for j in 0..ncand {
        let mut ttl = 0.0;
        for i in 0..ncit {
            ttl += net_scores[(i, j)];
        }
        utilities.push(ttl);
        if ttl > max_util {
            max_util = ttl;
        }
        avg_util += (ttl - avg_util) / ((j+1) as f64);
    }
    // Turn into regrets
    for u in utilities.iter_mut() {
        *u = (max_util - *u) / (max_util - avg_util);
    }
    utilities
}

pub fn elect_range_honest(net_scores: &Array2<f64>, ranks: u32) -> (Result, Result) {
    let (ncit, ncand) = net_scores.dim();
    let mut ttl_rankings = vec![0u32; ncand];
    for i in 0..ncit {
        range_score_honest(&mut ttl_rankings, &net_scores.subview(Axis(0), i), ranks);
    }
    println!("honest range<{}> votes: {:?}", ranks, &ttl_rankings);
    tally_votes(&ttl_rankings)
}

fn range_score_honest(ttl_rankings: &mut Vec<u32>, scores: &ArrayView<f64, Ix1>, ranks: u32) {
    let ncand = scores.dim();
    let mut min = scores[0];
    let mut max = scores[0];
    for j in 1..ncand {
        if scores[j] > max {
            max = scores[j];
        } else if scores[j] < min {
            min = scores[j];
        }
    }
    let rsz = (max - min) / ((ranks - 1) as f64);
    min -= rsz / 2.0; // a 0 is half a rank-size wide, as is ranks-1.
    for j in 0..ncand {
        let r = ((scores[j] - min) / rsz).floor() as u32;
        ttl_rankings[j] += r;
        //print!("   c{}: {:.2} = {}", j, scores[j], r);
    }
    //println!();
}

pub fn elect_range_strategic(net_scores: &Array2<f64>, ranks: u32, frac_strategic: f64,
        pre_results: &(Result, Result)) -> (Result, Result) {
    let (ncit, ncand) = net_scores.dim();
    let mut ttl_rankings = vec![0u32; ncand];
    let last_strategic = (ncit as f64 * frac_strategic).round() as usize;
    for i in 0..ncit {
        if i < last_strategic {
            // strategic voter. Favorite of top-two gets max score, other gets a zero.
            let mut max = net_scores[(i, pre_results.0.cand)];
            let mut min = net_scores[(i, pre_results.1.cand)];
            if max < min {
                let tmp = max;
                max = min;
                min = tmp;
            }
            let rsz = (max - min) / ((ranks - 1) as f64);
            min -= rsz / 2.0;
            for j in 0..ncand {
                let mut rf = ((net_scores[(i, j)] - min) / rsz).floor();
                if rf < 0.0 { rf = 0.0 }
                if rf > (ranks-1) as f64 { rf = (ranks-1) as f64 }
                let r = rf as u32;
                ttl_rankings[j] += r;
                //print!("   c{}: {:.2} = {}", j, net_scores[(i, j)], r);
            }
            //println!();
        } else {
            range_score_honest(&mut ttl_rankings, &net_scores.subview(Axis(0), i), ranks);
        }
    }
    println!("strategic range<{}> votes: {:?}", ranks, &ttl_rankings);
    tally_votes(&ttl_rankings)
}

pub fn rrv(net_scores: &Array2<f64>, ranks: u32, winners: u32) -> Vec<u32> {
    
}
