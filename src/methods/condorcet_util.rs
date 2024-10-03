/// Utilities for handling some Condorcet-criteria methods

use ndarray::Array2;
use crate::sim::Sim;

/// CandPair describes a pair-off for ranked systems: most voters rank the winner higher than the loser.
/// margin is the difference in votes: winner votes minus loser votes.
#[derive(Debug, Clone)]
pub struct CandPair {
    pub winner: usize,
    pub loser: usize,
    pub margin: i32,
}

pub fn find_candidate_pairoffs(pairs: &mut Vec<CandPair>, sim: &Sim) {
    pairs.clear();
    for icand in 0..sim.ncand {
        for jcand in (icand+1)..sim.ncand {
            if sim.i_beats_j_by[(icand, jcand)] > 0 {
                pairs.push(CandPair{
                    winner: icand,
                    loser: jcand,
                    margin: sim.i_beats_j_by[(icand, jcand)],
                });
            } else {
                pairs.push(CandPair{
                    winner: jcand,
                    loser: icand,
                    margin: -sim.i_beats_j_by[(icand, jcand)],
                });
            }
        }
    }
}

pub fn lock_in(locked_in: &mut Array2<bool>, pair: &CandPair, set: bool) {
    if pair.winner > pair.loser {
        locked_in[(pair.winner, pair.loser)] = set;
    } else {
        locked_in[(pair.loser, pair.winner)] = set;
    }
}

pub fn find_locked_in_winner(locked_in: &mut Array2<bool>, sim: &Sim) -> Option<usize> {
    'candidate: for iwin in 0..sim.ncand {
        let mut really_wins = false;
        let pi1 = (0..iwin).map(|i| (i, iwin));
        let pi2 = (iwin+1..sim.ncand).map(|j| (iwin, j));
        let pair_iter = pi1.chain(pi2);
        for (i, j) in pair_iter {
            if locked_in[(i, j)] {
                if sim.i_beats_j_by[(i, j)] > 0 {
                    really_wins = true; // But keep looking -- must beat all other locked-in pairs
                } else {
                    continue 'candidate // Not a winner
                }
            }
        }
        if really_wins {
            return Some(iwin);
        }
    }
    None // No candidate wins all races. Condorcet cycle.
}
