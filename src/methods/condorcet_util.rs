// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use crate::sim::Sim;
use log::*;
/// Utilities for handling some Condorcet-criteria methods
use ndarray::Array2;

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
        for jcand in (icand + 1)..sim.ncand {
            if sim.i_beats_j_by[(icand, jcand)] >= 0 {
                pairs.push(CandPair {
                    winner: icand,
                    loser: jcand,
                    margin: sim.i_beats_j_by[(icand, jcand)],
                });
            } else {
                pairs.push(CandPair {
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
        locked_in[(pair.loser, pair.winner)] = set; // (i, j) where j > i
    } else {
        locked_in[(pair.winner, pair.loser)] = set;
    }
}

pub fn find_locked_in_winner(locked_in: &mut Array2<bool>, sim: &Sim) -> Option<usize> {
    debug!("locked_in = {:?}", locked_in);
    'candidate: for iwin in 0..sim.ncand {
        let mut really_wins = false;
        let pi1 = (0..iwin).map(|i| (i, iwin, -1));
        let pi2 = (iwin + 1..sim.ncand).map(|j| (iwin, j, 1));
        let pair_iter = pi1.chain(pi2);
        for (i, j, i_is_iwin) in pair_iter {
            debug!(
                "iwin={}, i={}, j={}, locked_in? {}, i_beats_j_by={}",
                iwin,
                i,
                j,
                locked_in[(i, j)],
                sim.i_beats_j_by[(i, j)]
            );
            if locked_in[(i, j)] {
                if sim.i_beats_j_by[(i, j)] * i_is_iwin >= 0 {
                    really_wins = true; // But keep looking -- must beat all other locked-in pairs
                } else {
                    continue 'candidate; // Not a winner
                }
            }
        }
        if really_wins {
            return Some(iwin);
        }
    }
    None // No candidate wins all races. Condorcet cycle.
}

/// find_any_condorcet_winner returns the index of some one of the
/// Candidates in the Smith set. The Smith set is the smallest set
/// of candidates that beat all candidates not in the set in pairwise
/// elections.
/// This is used to initially seed the Smith set.
pub fn find_any_condorcet_winner(sim: &Sim) -> usize {
    let mut winner = usize::MAX; // invalid
    let mut max_victories = 0;
    for icand in 0..sim.ncand {
        let mut cand_victories = 0;
        for j in 0..sim.ncand {
            if j == icand {
                continue;
            }
            if sim.i_beats_j_by[(icand, j)] >= 0 {
                cand_victories += 1; // icand beats or ties with i
            }
        }
        if cand_victories >= max_victories {
            winner = icand;
            max_victories = cand_victories;
        }
    }
    winner
}

/// mark_smith_candidates is used by Sim. It's here because the code
/// is so closely related to the rest of this module.
pub fn mark_smith_candidates(sim: &mut Sim) {
    sim.in_smith_set.fill(false);
    let seed = find_any_condorcet_winner(sim);
    sim.in_smith_set[seed] = true;
    // Now include in the set all candidates which defeat one of the Smith candidates
    let mut icand = 0;
    'icand_loop: while icand < sim.ncand {
        if sim.in_smith_set[icand] {
            icand += 1;
            continue;
        }
        // icand is not yet in the Smith set. They become a member by defeating one who is.
        for j in 0..sim.ncand {
            if j == icand {
                continue;
            }
            if sim.in_smith_set[j] && sim.i_beats_j_by[(icand, j)] >= 0 {
                sim.in_smith_set[icand] = true; // icand beats or ties i
                icand = 0; // Must start from the top because there may be cycles.
                continue 'icand_loop;
            }
        }
        icand += 1;
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    // use super::*;
    use crate::methods::test_utils::sim_from_scores;
    use crate::sim::Sim;

    #[test]
    fn test_smith_set() {
        let mut sim = sim_from_scores(&[
            (&[-2., -3., -4., -1.], 40), // D>A>B>C
            (&[-3., -1., -2., -4.], 35), // B>C>A>D
            (&[-2., -3., -1., -4.], 25), // C>A>B>D
        ]);
        sim.rank_candidates(); // Gives us i_beats_j_by
        let expect_beats = array![
            [0, 40 - 35 + 25, 40 - 35 - 25, -40 + 35 + 25],
            [-30, 0, 40 + 35 - 25, -40 + 35 + 25],
            [20, -50, 0, -40 + 35 + 25],
            [-20, -20, -20, 0],
        ];
        assert_eq!(sim.i_beats_j_by, expect_beats);
        sim.find_smith_set();
        // (A>B, B>C, C>A), A>D, B>D, C>D
        assert_eq!(sim.in_smith_set, [true, true, true, false]);
        assert_eq!(sim.smith_set_size(), 3);

        let mut sim = Sim::new(4, 5);
        sim.scores = array![
            [-3.1, -4.1, -0.9, -3.3],
            [-5.2, -2.2, -2.9, -5.3],
            [-3.5, -4.0, -1.0, -3.7],
            [-5.4, -3.1, -2.8, -5.6],
            [-2.1, -6.0, -0.8, -2.5]
        ];
        sim.rank_candidates();
        #[rustfmt::skip]
        assert_eq!(
            sim.i_beats_j_by,
            array![
                [ 0, 1, -5,  5],
                [-1, 0, -3, -1],
                [ 5, 3,  0,  5],
                [-5, 1, -5,  0],
            ]
        );
        sim.find_smith_set();
        assert_eq!(sim.smith_set_size(), 1);
        assert_eq!(sim.in_smith_set, [false, false, true, false]);
    }
}
