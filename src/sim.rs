use crate::considerations::*;
use crate::methods::condorcet_util::mark_smith_candidates;
use crate::methods::{ElectResult, WinnerAndRunnerup};
use ndarray::{Array2, Axis};
use rand::rngs::ThreadRng;

pub struct Sim {
    pub ncand: usize,
    pub ncit: usize,
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
    pub fn new(ncand: usize, ncit: usize) -> Sim {
        Sim {
            ncand,
            ncit,
            scores: Array2::zeros((ncit, ncand)),
            i_beats_j_by: Array2::zeros((ncand, ncand)),
            ranks: Array2::zeros((ncit, ncand)),
            regrets: vec![0.0; ncand],
            cand_by_regret: (0..ncand).collect(),
            regret_rank: (0..ncand).collect(),
            in_smith_set: vec![false; ncand],
            scratch_ranks: (0..ncand).collect(),
        }
    }

    pub fn election(
        &mut self,
        axes: &mut [Box<dyn ConsiderationSim>],
        rng: &mut ThreadRng,
        verbose: bool,
    ) {
        self.get_scores(axes, rng, verbose);
        self.compute_regrets();
        self.rank_candidates();
        self.find_smith_set();
    }

    pub fn take_from_primary(&mut self, primary: &Sim, winners: &[ElectResult]) {
        assert!(primary.ncit == self.ncit);
        assert!(winners.len() == self.ncand);
        for (icand, winner) in winners.iter().enumerate() {
            for icit in 0..self.ncit {
                self.scores[(icit, icand)] = primary.scores[(icit, winner.cand)];
            }
        }
        self.compute_regrets();
        self.rank_candidates();
    }

    fn get_scores(
        &mut self,
        axes: &mut [Box<dyn ConsiderationSim>],
        rng: &mut ThreadRng,
        verbose: bool,
    ) {
        self.scores.fill(0.0);
        for ax in axes.iter_mut() {
            ax.add_to_scores(&mut self.scores, rng, verbose);
        }
        if self.ncit < 20 && verbose {
            println!("Voter utilities:\n{:?}", &mut self.scores);
        }
    }

    // Side-effects: compute self.regrets and self.cand_by_regret
    pub fn compute_regrets(&mut self) {
        let mut max_util = f64::MIN;
        let mut avg_util = 0.0;
        for j in 0..self.ncand {
            let mut ttl = 0.0;
            for i in 0..self.ncit {
                ttl += self.scores[(i, j)];
            }
            self.regrets[j] = ttl;
            if ttl > max_util {
                max_util = ttl;
            }
            avg_util += (ttl - avg_util) / ((j + 1) as f64);
        }
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
        // for i in 0..self.ncit {
        self.i_beats_j_by.fill(0);
        for (icit, cit_scores) in self.scores.axis_iter(Axis(0)).enumerate() {
            self.scratch_ranks
                .sort_by(|&a, &b| cit_scores[b].partial_cmp(&cit_scores[a]).unwrap());
            for icand in 0..self.ncand {
                self.ranks[(icit, icand)] = self.scratch_ranks[icand];
                for jcand in 0..icand {
                    if cit_scores[icand] > cit_scores[jcand] {
                        self.i_beats_j_by[(icand, jcand)] += 1;
                    } else if cit_scores[icand] < cit_scores[jcand] {
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
            result.clone()
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
                result.clone()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
