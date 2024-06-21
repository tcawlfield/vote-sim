// Voting methods
use ndarray::{Array2, ArrayView, Axis, Ix1};
use rand::Rng;
use std::f64;

#[derive(Debug)]
pub struct ElectResult {
    pub cand: usize,
    pub score: f64,
}

pub fn elect_plurality_honest(
    net_scores: &Array2<f64>,
    verbose: bool,
) -> (ElectResult, ElectResult) {
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
    if verbose {
        println!("honest plurality votes: {:?}", votes);
    }
    tally_votes(&votes)
}

fn tally_votes(votes: &Vec<u32>) -> (ElectResult, ElectResult) {
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
    if most_votes == runup_votes {
        if rand::random() {
            // 50/50 chance
            (electee, runup) = (runup, electee);
            (most_votes, runup_votes) = (runup_votes, most_votes);
        }
    }
    // TODO: We still are flubbing cases where there's a tie for runner-up or 3+-way for 1st.
    (
        ElectResult {
            cand: electee,
            score: most_votes as f64,
        },
        ElectResult {
            cand: runup,
            score: runup_votes as f64,
        },
    )
}

pub fn elect_plurality_strategic(
    net_scores: &Array2<f64>,
    frac_strategic: f64,
    pre_results: &(ElectResult, ElectResult),
    verbose: bool,
) -> (ElectResult, ElectResult) {
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
    if verbose {
        println!("strategic plurality votes: {:?}", votes);
    }
    tally_votes(&votes)
}

pub fn elect_range_honest(
    net_scores: &Array2<f64>,
    ranks: u32,
    verbose: bool,
) -> (ElectResult, ElectResult) {
    let (ncit, ncand) = net_scores.dim();
    let mut ttl_rankings = vec![0u32; ncand];
    for i in 0..ncit {
        range_score_honest(
            ttl_rankings.as_mut_slice(),
            &net_scores.index_axis(Axis(0), i),
            ranks,
        );
    }
    if verbose {
        println!("honest range<{}> votes: {:?}", ranks, &ttl_rankings);
    }
    tally_votes(&ttl_rankings)
}

fn range_score_honest(ttl_rankings: &mut [u32], scores: &ArrayView<f64, Ix1>, ranks: u32) {
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

pub fn elect_range_strategic(
    net_scores: &Array2<f64>,
    ranks: u32,
    frac_strategic: f64,
    pre_results: &(ElectResult, ElectResult),
    verbose: bool,
) -> (ElectResult, ElectResult) {
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
                if rf < 0.0 {
                    rf = 0.0
                }
                if rf > (ranks - 1) as f64 {
                    rf = (ranks - 1) as f64
                }
                let r = rf as u32;
                ttl_rankings[j] += r;
                //print!("   c{}: {:.2} = {}", j, net_scores[(i, j)], r);
            }
            //println!();
        } else {
            range_score_honest(
                ttl_rankings.as_mut_slice(),
                &net_scores.index_axis(Axis(0), i),
                ranks,
            );
        }
    }
    if verbose {
        println!("strategic range<{}> votes: {:?}", ranks, &ttl_rankings);
    }
    tally_votes(&ttl_rankings)
}

// K = 1.0 favors large political parties. K = 0.5 favors smaller parties (more penalty).
// I'm using this purely as a method of spreading out candidates across the position axes.
const K: f64 = 0.5;

// Reweighted Ranve Voting -- a system for proportional representation.
// See http://rangevoting.org/RRV.html
pub fn rrv(net_scores: &Array2<f64>, ranks: u32, num_winners: usize) -> Vec<usize> {
    let (ncit, ncand) = net_scores.dim();
    let mut remaining = (0..ncand).collect::<Vec<usize>>();
    let mut winners = Vec::with_capacity(num_winners as usize);
    let mut score_cards: Array2<u32> = Array2::zeros((ncit, ncand));
    for i in 0..ncit {
        range_score_honest(
            score_cards
                .index_axis_mut(Axis(0), i)
                .as_slice_mut()
                .unwrap(),
            &net_scores.index_axis(Axis(0), i),
            ranks,
        );
    }
    //println!("{:?}", score_cards);

    while winners.len() < num_winners {
        //println!("Round {}", winners.len());
        let mut ttl_scores = vec![0.0; ncand];
        for i in 0..ncit {
            // Weight is K / (K + SUM/MAX)
            let sum = winners.iter().fold(0, |sum, &j| sum + score_cards[(i, j)]);
            let wt = K / (K + (sum as f64) / ((ranks - 1) as f64));
            for j in remaining.iter() {
                ttl_scores[*j] += wt * (score_cards[(i, *j)] as f64);
            }
        }
        //println!("ttl_scores = {:?}", ttl_scores);
        // let winner = remaining.iter()
        //                       .max_by_key(|&j| ttl_scores[*j]).unwrap();
        // let winner_idx = remaining.iter().find(|&j| j == winner).unwrap();
        let winner_idx = {
            let mut rem_iter = remaining.iter();
            let mut winner_idx = 0;
            let mut winner_score = ttl_scores[*rem_iter.next().unwrap()];
            //println!("     winner = {}, score = {}", remaining[winner_idx], winner_score);
            for (idx, j) in rem_iter.enumerate() {
                if ttl_scores[*j] > winner_score {
                    winner_idx = idx + 1;
                    winner_score = ttl_scores[*j];
                    //println!("     New winner={}, score={}", remaining[winner_idx], winner_score);
                }
            }
            winner_idx
        };
        winners.push(remaining.swap_remove(winner_idx));
        //println!("winners = {:?}, remaining = {:?}", winners, remaining);
    }

    winners
}

pub fn get_ranked_ballots(net_scores: &Array2<f64>) -> Array2<usize> {
    let (ncit, ncand) = net_scores.dim();
    let mut ranks = Array2::zeros((ncit, ncand));
    let mut my_ranks = Vec::with_capacity(ncand);
    for j in 0..ncand {
        my_ranks.push(j);
    }
    for i in 0..ncit {
        // my_ranks.sort_unstable_by(|a, b| net_scores[(i,b)].cmp(net_scores[(i,a)]));
        my_ranks.sort_by(|&a, &b| net_scores[(i, b)].partial_cmp(&net_scores[(i, a)]).unwrap());
        for j in 0..ncand {
            ranks[(i, j)] = my_ranks[j];
        }
    }
    ranks
}

pub fn elect_irv_honest(ballots: &Array2<usize>, verbose: bool) -> Option<usize> {
    let (ncit, ncand) = ballots.dim();
    let mut votes = vec![0; ncand];
    let mut eliminated = vec![false; ncand];
    loop {
        if verbose {
            println!("IRV round: eliminated = {:?}", eliminated);
        }
        let mut active_voters = 0;
        for j in 0..ncand {
            votes[j] = 0;
        }
        for i in 0..ncit {
            let mut best: Option<usize> = None; // no top candidate yet
            for j in 0..ncand {
                let ranked_next = ballots[(i, j)];
                // if !eliminated.iter().any(|&k| k == ranked_next) {
                if !eliminated[ranked_next] {
                    best = Some(ranked_next);
                    break;
                }
            }
            if let Some(j) = best {
                active_voters += 1;
                votes[j] += 1;
            }
        }
        if verbose {
            println!("  tallies are: {:?}", votes);
        }
        let mut best = ncand; // invalid index
        let mut worst = ncand;
        for j in 0..ncand {
            if !eliminated[j] {
                if best == ncand {
                    best = j;
                } else {
                    worst = j;
                    break;
                }
            }
        }
        if worst == ncand {
            return None;
        }
        let start_idx = worst;
        if votes[worst] > votes[best] {
            worst = best;
            best = start_idx;
        }
        let mut best_votes = votes[best];
        let mut worst_votes = votes[worst];
        for j in (start_idx + 1)..ncand {
            if !eliminated[j] {
                if votes[j] > best_votes {
                    best_votes = votes[j];
                    best = j;
                } else if votes[j] < worst_votes {
                    worst_votes = votes[j];
                    worst = j;
                }
            }
        }
        if best_votes >= (active_voters + 1) / 2 {
            return Some(best);
        } else {
            // We have a run-off!
            eliminated[worst] = true;
        }
    }
}

pub fn elect_borda_honest(
    net_scores: &Array2<f64>,
    ballots: &Array2<usize>,
    top_n: Option<usize>,
    verbose: bool,
) -> (ElectResult, ElectResult) {
    let (ncit, ncand) = ballots.dim();
    let top_n = if let Some(n) = top_n { n } else { ncand };
    let mut points: Vec<u32> = vec![0; ncand];
    for icit in 0..ncit {
        for icandrank in 0..top_n {
            points[ballots[(icit, icandrank)]] += (top_n - icandrank) as u32;
        }
    }
    if verbose {
        println!("Borda point totals for top {}: {:?}", top_n, points);
    }
    tally_votes_with_plurality_for_ties(&points, &net_scores, verbose)
}

fn tally_votes_with_plurality_for_ties(
    votes: &Vec<u32>,
    net_scores: &Array2<f64>,
    verbose: bool,
) -> (ElectResult, ElectResult) {
    let (ncit, ncand) = net_scores.dim();
    let mut ntop: usize = 1;
    let mut top_cands: Vec<usize> = vec![0; ncand];
    let mut most_votes = votes[0];

    let mut nsecond: usize = 0;
    let mut second_cands: Vec<usize> = vec![0; ncand];
    let mut runup_votes: u32 = 0;

    for icand in 1..ncand {
        if votes[icand] > most_votes {
            // New best
            nsecond = ntop; // Previous best becomes runner-up
            for i in 0..ntop {
                second_cands[i] = top_cands[i];
            }
            runup_votes = most_votes;

            ntop = 1;
            top_cands[0] = icand;
            most_votes = votes[icand];
        } else if votes[icand] == most_votes {
            // new tie for top
            top_cands[ntop] = icand;
            ntop += 1;
        } else if votes[icand] > runup_votes {
            // new runner-up candidate
            nsecond = 1;
            second_cands[0] = icand;
            runup_votes = votes[icand];
        } else if votes[icand] == runup_votes {
            // new tie for runner-up
            second_cands[nsecond] = icand;
            nsecond += 1;
        }
    }
    if ntop > 1 {
        // Do a runoff
        let mut some_scores = Array2::zeros((ncit, ntop));
        for icit in 0..ncit {
            for icand in 0..ntop {
                some_scores[(icit, icand)] = net_scores[(icit, top_cands[icand])];
            }
        }
        let mut runoff_results = elect_plurality_honest(&some_scores, verbose);
        runoff_results.0.cand = top_cands[runoff_results.0.cand];
        runoff_results.1.cand = top_cands[runoff_results.1.cand];
        runoff_results
    } else if nsecond > 1 {
        // Don't bother to do a runoff.
        let mut rng = rand::thread_rng();
        let chosen_second = rng.gen_range(0..nsecond);
        (
            ElectResult {
                cand: top_cands[0],
                score: most_votes as f64,
            },
            ElectResult {
                cand: second_cands[chosen_second],
                score: runup_votes as f64,
            },
        )
    } else if nsecond == 0 {
        (
            ElectResult {
                cand: top_cands[0],
                score: most_votes as f64,
            },
            ElectResult {
                cand: 0,
                score: -1.0,
            },
        )
    } else {
        (
            ElectResult {
                cand: top_cands[0],
                score: most_votes as f64,
            },
            ElectResult {
                cand: second_cands[0],
                score: runup_votes as f64,
            },
        )
    }
}

pub fn elect_range_honest_with_tie_runoff(
    net_scores: &Array2<f64>,
    ranks: u32,
    verbose: bool,
) -> (ElectResult, ElectResult) {
    let (ncit, ncand) = net_scores.dim();
    let mut ttl_rankings = vec![0u32; ncand];
    for i in 0..ncit {
        range_score_honest(
            ttl_rankings.as_mut_slice(),
            &net_scores.index_axis(Axis(0), i),
            ranks,
        );
    }
    if verbose {
        println!("honest range<{}> votes: {:?}", ranks, &ttl_rankings);
    }
    tally_votes_with_plurality_for_ties(&ttl_rankings, &net_scores, verbose)
}

pub fn elect_multivoting_with_tie_runoff(
    net_scores: &Array2<f64>,
    votes: u32,
    spread_fact: f64,
    verbose: bool,
) -> (ElectResult, ElectResult) {
    let (ncit, ncand) = net_scores.dim();
    let mut ttl_points = vec![0u32; ncand];
    let mut cand_scores = vec![0f64; ncand];
    for icit in 0..ncit {
        let mut ttl_score = 0.0;
        let mut min_score = net_scores[(icit, 0)];
        for icand in 0..ncand {
            cand_scores[icand] = net_scores[(icit, icand)];
            ttl_score += cand_scores[icand];
            if cand_scores[icand] < min_score {
                min_score = cand_scores[icand];
            }
        }
        let reduction = spread_fact * (ttl_score - min_score * ncand as f64) / (votes as f64);
        // if verbose {
        //     println!("  voter {} has reduction {}", icit, reduction);
        // }
        for _ in 0..votes {
            let mut max_score = cand_scores[0];
            let mut best_cand = 0;
            for i in 1..ncand {
                if cand_scores[i] > max_score {
                    best_cand = i;
                    max_score = cand_scores[i];
                }
            }
            ttl_points[best_cand] += 1;
            cand_scores[best_cand] -= reduction;
            // if verbose {
            //     println!(
            //         "Voter {} votes for {}, scores are now {:?}",
            //         icit, best_cand, cand_scores
            //     );
            // }
        }
    }
    if verbose {
        println!("Multivoting vote totals: {:?}", ttl_points);
    }
    tally_votes_with_plurality_for_ties(&ttl_points, &net_scores, verbose)
}
