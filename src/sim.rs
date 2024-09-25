use crate::considerations::*;
use crate::methods::{ElectResult, WinnerAndRunnerup};
use ndarray::{Array2, Axis};
use rand::rngs::ThreadRng;

pub struct Sim {
    pub ncand: usize,
    pub ncit: usize,
    pub scores: Array2<f64>,
    pub ranks: Array2<usize>,
    pub regrets: Vec<f64>,
    pub cand_by_regret: Vec<usize>,
    scratch_ranks: Vec<usize>,
}

impl Sim {
    pub fn new(ncand: usize, ncit: usize) -> Sim {
        let scores: Array2<f64> = Array2::zeros((ncit, ncand));
        let ranks: Array2<usize> = Array2::zeros((ncit, ncand));
        Sim {
            ncand,
            ncit,
            scores,
            ranks,
            regrets: vec![0.0; ncand],
            cand_by_regret: (0..ncand).collect(),
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
        self.cand_by_regret
            .sort_by(|&a, &b| self.regrets[a].partial_cmp(&self.regrets[b]).unwrap_or(std::cmp::Ordering::Equal));
    }

    pub fn rank_candidates(&mut self) {
        for i in 0..self.ncit {
            self.scratch_ranks.sort_by(|&a, &b| {
                self.scores[(i, b)]
                    .partial_cmp(&self.scores[(i, a)])
                    .unwrap()
            });
            for j in 0..self.ncand {
                self.ranks[(i, j)] = self.scratch_ranks[j];
            }
        }
    }

    pub fn break_tie_with_plurality(&self, result: &WinnerAndRunnerup) -> WinnerAndRunnerup {
        if !result.is_tied() {
            result.clone()
        } else {
            let mut runup_votes = 0;
            let mut winner_votes = 0;
            for utilities in self.scores.axis_iter(Axis(0)) {
                if utilities[result.winner.cand] >= utilities[result.runnerup.cand] {
                    winner_votes += 1;
                } else {
                    runup_votes += 1;
                }
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

    // pub fn reduced_with_rrv(orig_sim: &Sim, ncand_final: usize) -> Sim {
    // }
}
