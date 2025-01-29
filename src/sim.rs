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
    pub regret_rank: Vec<usize>, // map icand to regret-ranked pos'n
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
                for jcand in (icand + 1)..self.ncand {
                    if cit_scores[icand] > cit_scores[jcand] {
                        self.i_beats_j_by[(icand, jcand)] += 1;
                    }
                }
            }
        }

        for ((i, j), beat_by) in self.i_beats_j_by.indexed_iter_mut() {
            if j > i {
                // Only upper-triangular elements are used -- or lower-triangular?
                *beat_by = 2 * *beat_by - self.ncit as i32; // total ordering, so defeats = ncit - n_i_beats_j.
            }
        }
    }

    /// find_smith_set fills in in_smith_set array.
    /// Requires rand_candidates to have been called.
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
