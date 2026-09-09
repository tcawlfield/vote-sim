// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use crate::considerations::*;
use crate::methods::condorcet_util::mark_smith_candidates;
use crate::methods::{ElectResult, WinnerAndRunnerup};
use ndarray::{Array2, Axis};
use rand::Rng;

pub struct Sim {
    pub ncand: usize,
    pub nvtr: usize,
    pub scores: Array2<f64>,
    pub ranks: Array2<usize>,
    pub i_beats_j_by: Array2<i32>,
    pub regrets: Vec<f64>,
    pub cand_by_regret: Vec<usize>, // map from regret rank to icand
    pub regret_rank: Vec<usize>,    // map icand to regret-ranked pos'n
    pub in_smith_set: Vec<bool>,
    scratch_ranks: Vec<usize>,
}

impl Sim {
    pub fn new(ncand: usize, nvtr: usize) -> Sim {
        Sim {
            ncand,
            nvtr,
            scores: Array2::zeros((nvtr, ncand)),
            i_beats_j_by: Array2::zeros((ncand, ncand)),
            ranks: Array2::zeros((nvtr, ncand)),
            regrets: vec![0.0; ncand],
            cand_by_regret: (0..ncand).collect(),
            regret_rank: (0..ncand).collect(),
            in_smith_set: vec![false; ncand],
            scratch_ranks: (0..ncand).collect(),
        }
    }

    pub fn election<R: Rng + ?Sized>(&mut self, axes: &mut [ConsiderationSimKind], rng: &mut R) {
        self.get_scores(axes, rng);
        self.compute_regrets();
        self.rank_candidates();
        self.find_smith_set();
    }

    pub fn take_from_primary(&mut self, primary: &Sim, winners: &[ElectResult]) {
        assert!(primary.nvtr == self.nvtr);
        assert!(winners.len() == self.ncand);
        for (icand, winner) in winners.iter().enumerate() {
            self.scores
                .column_mut(icand)
                .assign(&primary.scores.column(winner.cand));
        }
        self.compute_regrets();
        self.rank_candidates();
        self.find_smith_set();
    }

    fn get_scores<R: Rng + ?Sized>(&mut self, axes: &mut [ConsiderationSimKind], rng: &mut R) {
        self.scores.fill(0.0);
        for ax in axes.iter_mut() {
            ax.add_to_scores(&mut self.scores, rng);
        }
        log::debug!("Voter utilities:\n{:?}", self.scores);
    }

    // Side-effects: compute self.regrets and self.cand_by_regret
    pub fn compute_regrets(&mut self) {
        // Total each candidate's utility across all voters, accumulating into the
        // reused self.regrets buffer (contiguous traversal, no allocation).
        self.regrets.fill(0.0);
        for voter_scores in self.scores.rows() {
            for (regret, s) in self.regrets.iter_mut().zip(voter_scores) {
                *regret += s;
            }
        }
        let max_util = self.regrets.iter().copied().fold(f64::MIN, f64::max);
        let avg_util = self.regrets.iter().sum::<f64>() / self.ncand as f64;
        // Turn into regrets
        for u in self.regrets.iter_mut() {
            *u = (max_util - *u) / (max_util - avg_util);
        }
        self.cand_by_regret.sort_by(|&a, &b| {
            self.regrets[a]
                .partial_cmp(&self.regrets[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (irr, &icand) in self.cand_by_regret.iter().enumerate() {
            self.regret_rank[icand] = irr;
        }
    }

    /// rank_candidates uses the score table to fix the table of
    /// candidate rankings (Sim.ranks), and also fills in the i_beats_j_by matrix.
    pub fn rank_candidates(&mut self) {
        // for i in 0..self.nvtr {
        self.i_beats_j_by.fill(0);
        for (ivtr, vtr_scores) in self.scores.axis_iter(Axis(0)).enumerate() {
            self.scratch_ranks
                .sort_by(|&a, &b| vtr_scores[b].partial_cmp(&vtr_scores[a]).unwrap());
            for icand in 0..self.ncand {
                self.ranks[(ivtr, icand)] = self.scratch_ranks[icand];
                for jcand in 0..icand {
                    if vtr_scores[icand] > vtr_scores[jcand] {
                        self.i_beats_j_by[(icand, jcand)] += 1;
                    } else if vtr_scores[icand] < vtr_scores[jcand] {
                        // This is a slowdown, but handles equal-score cases (which should be nearly nonexistent)
                        self.i_beats_j_by[(jcand, icand)] += 1;
                    }
                }
            }
        }

        // i_beats_j_by is misnamed until we convert to a margin of victory
        for i in 1..self.ncand {
            for j in 0..i {
                let ibj = self.i_beats_j_by[(i, j)];
                self.i_beats_j_by[(i, j)] -= self.i_beats_j_by[(j, i)];
                self.i_beats_j_by[(j, i)] -= ibj;
            }
        }
    }

    /// find_smith_set fills in in_smith_set array.
    /// Requires rank_candidates to have been called.
    pub fn find_smith_set(&mut self) {
        mark_smith_candidates(self);
    }

    /// Returns the size of the Smith set.
    /// Requires find_smith_set to have been called.
    pub fn smith_set_size(&self) -> usize {
        self.in_smith_set.iter().filter(|b| **b).count()
    }

    /// If winner and runnerup have the same score, break_tie_with_plurality
    /// will swap runnerup and winner if the runnerup would win a plurality vote.
    pub fn break_tie_with_plurality(&self, result: &WinnerAndRunnerup) -> WinnerAndRunnerup {
        if !result.is_tied() {
            *result
        } else {
            let mut runup_votes = 0;
            let mut winner_votes = 0;
            for utilities in self.scores.axis_iter(Axis(0)) {
                if utilities[result.winner.cand] > utilities[result.runnerup.cand] {
                    winner_votes += 1;
                } else if utilities[result.winner.cand] < utilities[result.runnerup.cand] {
                    runup_votes += 1;
                }
                // Equal scores don't count. But we don't usually get exactly equal scores.
            }
            if runup_votes > winner_votes {
                WinnerAndRunnerup {
                    winner: result.runnerup,
                    runnerup: result.winner,
                }
            } else {
                *result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_ranks_beats_and_smith() {
        // Example from https://electowiki.org/wiki/Ranked_Pairs#Notes
        let mut sim = Sim::new(6, 5);
        sim.scores = array![
            [-1., -2., -3., -4., -5., -6.],
            [-1., -2., -3., -4., -5., -6.],
            [-2., -3., -1., -5., -6., -4.],
            [-3., -1., -2., -6., -4., -5.],
            [-1., -2., -1., -4., -5., -4.],
        ];
        sim.rank_candidates(); // Creates the i_beats_j matrix in sim
        #[rustfmt::skip]
        assert_eq!(sim.ranks, array![
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [2, 0, 1, 5, 3, 4], // cand 2 is ranked 1st, etc.
            [1, 2, 0, 4, 5, 3],
            [2, 0, 1, 5, 3, 4], // stable sort of 0, 1, 3, 4 from row above
        ]);
        #[rustfmt::skip]
        assert_eq!(sim.i_beats_j_by, array![
            [ 0,  3,  0,  5,  5,  5], // i goes down, j goes across. j > i.
            [-3,  0,  1,  5,  5,  5],
            [ 0, -1,  0,  5,  5,  5],
            [-5, -5, -5,  0,  3,  0],
            [-5, -5, -5, -3,  0,  1],
            [-5, -5, -5,  0, -1,  0],
        ]);
        sim.find_smith_set();
        assert_eq!(
            sim.in_smith_set,
            vec![true, true, true, false, false, false]
        );
    }

    #[test]
    fn new_sim_has_expected_shapes_and_identity_orderings() {
        let sim = Sim::new(3, 5);
        assert_eq!(sim.scores.dim(), (5, 3));
        assert_eq!(sim.ranks.dim(), (5, 3));
        assert_eq!(sim.i_beats_j_by.dim(), (3, 3));
        assert_eq!(sim.regrets, vec![0.0; 3]);
        assert_eq!(sim.cand_by_regret, vec![0, 1, 2]);
        assert_eq!(sim.regret_rank, vec![0, 1, 2]);
        assert_eq!(sim.in_smith_set, vec![false; 3]);
    }

    #[test]
    fn compute_regrets_normalizes_best_to_zero_and_average_to_one() {
        let mut sim = Sim::new(3, 4);
        // Per-candidate utility totals: A = 12, B = 6, C = 0  ->  average = 6.
        sim.scores = array![[3., 2., 0.], [3., 1., 0.], [3., 2., 0.], [3., 1., 0.]];
        sim.compute_regrets();
        // regret = (max - total) / (max - avg) = (12 - total) / 6
        assert_abs_diff_eq!(sim.regrets[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sim.regrets[1], 1.0, epsilon = 1e-12); // the average candidate
        assert_abs_diff_eq!(sim.regrets[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn compute_regrets_orders_candidates_by_ascending_regret() {
        let mut sim = Sim::new(4, 3);
        // Totals: cand0 = 6, cand1 = 3, cand2 = 9, cand3 = 0.
        sim.scores = array![[2., 1., 3., 0.], [2., 1., 3., 0.], [2., 1., 3., 0.]];
        sim.compute_regrets();

        assert_eq!(sim.cand_by_regret, vec![2, 0, 1, 3]);
        assert_abs_diff_eq!(sim.regrets[2], 0.0, epsilon = 1e-12); // best candidate has zero regret

        // regret_rank is the inverse permutation of cand_by_regret
        for (rank, &icand) in sim.cand_by_regret.iter().enumerate() {
            assert_eq!(sim.regret_rank[icand], rank);
        }
        // regrets are non-decreasing when walked in cand_by_regret order
        let ordered: Vec<f64> = sim.cand_by_regret.iter().map(|&c| sim.regrets[c]).collect();
        assert!(ordered.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn take_from_primary_copies_winner_columns_and_recomputes() {
        let mut primary = Sim::new(4, 3);
        primary.scores = array![
            [10., 20., 30., 40.],
            [11., 21., 31., 41.],
            [12., 22., 32., 42.],
        ];
        // A 2-candidate final field: primary winners are candidates 3 and 1.
        let mut sim = Sim::new(2, 3);
        let winners = [
            ElectResult {
                cand: 3,
                score: 0.0,
            },
            ElectResult {
                cand: 1,
                score: 0.0,
            },
        ];
        sim.take_from_primary(&primary, &winners);

        assert_eq!(sim.scores, array![[40., 20.], [41., 21.], [42., 22.]]);
        // Totals 123 and 63, avg 93 -> regrets (123 - total) / 30.
        assert_abs_diff_eq!(sim.regrets[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sim.regrets[1], 2.0, epsilon = 1e-12);
        assert_eq!(sim.cand_by_regret, vec![0, 1]);
    }

    #[test]
    #[should_panic]
    fn take_from_primary_rejects_voter_count_mismatch() {
        let primary = Sim::new(3, 5);
        let mut sim = Sim::new(2, 4);
        sim.take_from_primary(
            &primary,
            &[
                ElectResult { cand: 0, score: 0. },
                ElectResult { cand: 1, score: 0. },
            ],
        );
    }

    #[test]
    #[should_panic]
    fn take_from_primary_rejects_wrong_winner_count() {
        let primary = Sim::new(3, 4);
        let mut sim = Sim::new(2, 4);
        sim.take_from_primary(&primary, &[ElectResult { cand: 0, score: 0. }]);
    }

    #[test]
    fn rank_candidates_unanimous_preference_gives_antisymmetric_full_margins() {
        let mut sim = Sim::new(3, 4);
        // Every voter ranks cand 2 > cand 0 > cand 1.
        sim.scores = array![[2., 1., 3.], [2., 1., 3.], [2., 1., 3.], [2., 1., 3.]];
        sim.rank_candidates();

        for row in sim.ranks.rows() {
            assert_eq!(row.to_vec(), vec![2, 0, 1]);
        }
        // Margins are antisymmetric, and unanimous pairings are +/- nvtr.
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(sim.i_beats_j_by[(i, j)], -sim.i_beats_j_by[(j, i)]);
            }
        }
        assert_eq!(sim.i_beats_j_by[(2, 0)], 4);
        assert_eq!(sim.i_beats_j_by[(2, 1)], 4);
        assert_eq!(sim.i_beats_j_by[(0, 1)], 4);
    }

    fn tied(winner: usize, runnerup: usize) -> WinnerAndRunnerup {
        WinnerAndRunnerup {
            winner: ElectResult {
                cand: winner,
                score: 9.0,
            },
            runnerup: ElectResult {
                cand: runnerup,
                score: 9.0,
            },
        }
    }

    #[test]
    fn break_tie_returns_untied_result_unchanged() {
        let sim = Sim::new(3, 2);
        let r = WinnerAndRunnerup {
            winner: ElectResult {
                cand: 0,
                score: 5.0,
            },
            runnerup: ElectResult {
                cand: 1,
                score: 3.0,
            },
        };
        let out = sim.break_tie_with_plurality(&r);
        assert_eq!((out.winner.cand, out.runnerup.cand), (0, 1));
    }

    #[test]
    fn break_tie_swaps_when_runnerup_wins_the_head_to_head() {
        let mut sim = Sim::new(2, 3);
        // 2 of 3 voters score candidate 1 above candidate 0.
        sim.scores = array![[1., 5.], [1., 5.], [5., 1.]];
        let out = sim.break_tie_with_plurality(&tied(0, 1));
        assert_eq!((out.winner.cand, out.runnerup.cand), (1, 0));
    }

    #[test]
    fn break_tie_keeps_order_when_winner_wins_the_head_to_head() {
        let mut sim = Sim::new(2, 3);
        sim.scores = array![[5., 1.], [5., 1.], [1., 5.]];
        let out = sim.break_tie_with_plurality(&tied(0, 1));
        assert_eq!(out.winner.cand, 0);
    }

    #[test]
    fn break_tie_keeps_order_on_a_head_to_head_tie() {
        let mut sim = Sim::new(2, 2);
        sim.scores = array![[5., 1.], [1., 5.]];
        let out = sim.break_tie_with_plurality(&tied(0, 1));
        assert_eq!(out.winner.cand, 0); // swap needs runnerup strictly ahead
    }

    #[test]
    fn election_with_likability_only_is_unanimous() {
        use crate::considerations::{Consideration, Likability};

        let mut sim = Sim::new(4, 20);
        let mut axes = vec![Consideration::Likability(Likability { mean: 1.0 }).new_sim(&sim)];
        sim.election(&mut axes, &mut rand::rng());

        // Likability is shared by all voters, so there is a strict Condorcet
        // winner and the Smith set holds exactly one candidate.
        assert_eq!(sim.smith_set_size(), 1);
        let best = sim.cand_by_regret[0];
        assert!(sim.in_smith_set[best]);
        assert_abs_diff_eq!(sim.regrets[best], 0.0, epsilon = 1e-12);

        // Unanimous: every voter produces the same ranking.
        let first = sim.ranks.row(0).to_vec();
        for row in sim.ranks.rows() {
            assert_eq!(row.to_vec(), first);
        }
    }
}
