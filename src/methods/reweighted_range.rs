use ndarray::{Array2, Axis};

use super::method::{Strategy, WinnerAndRunnerup};
use crate::sim::Sim;
use crate::ElectResult;
use super::rangevoting::fill_range_ballot;

#[derive(Debug)]
pub struct RRV {
    pub strat: Strategy,
    ranks: i32,
    wtd_scores: Vec<f64>,
    ballots: Array2<i32>,
    winners: Vec<ElectResult>,
    remaining: Vec<usize>,
}

// K = 1.0 favors large political parties. K = 0.5 favors smaller parties (more penalty).
// I'm using this purely as a method of spreading out candidates across the position axes.
const K: f64 = 0.5;

impl RRV {
    pub fn new(sim: &Sim, ranks: i32, strat: Strategy) -> RRV {
        RRV {
            strat,
            ranks,
            wtd_scores: vec![0.; sim.ncand],
            ballots: Array2::zeros((sim.ncit, sim.ncand)),
            winners: Vec::with_capacity(sim.ncand),
            remaining: Vec::with_capacity(sim.ncand),
        }
    }

    pub fn multi_elect(
        &mut self,
        sim: &Sim,
        _honest_rslt: Option<WinnerAndRunnerup>,
        nwinners: usize,
        _verbose: bool,
    ) -> &Vec<ElectResult> {
        self.ballots.fill(0);
        for icit in 0..sim.ncit {
            fill_range_ballot(
                &sim.scores.index_axis(Axis(0), icit),
                self.ranks,
                self.ballots.index_axis_mut(Axis(0), icit).as_slice_mut().unwrap()
            );
        }

        self.remaining.clear();
        self.remaining.extend(0..sim.ncand);
        self.winners.clear();
        while self.winners.len() < nwinners {
            self.wtd_scores.fill(0.0);
            let ttl_scores = vec![0.0; sim.ncand];
            for i in 0..sim.ncit {
                // Weight is K / (K + SUM/MAX)
                let sum = self.winners.iter().fold(0, |sum, j| sum + self.ballots[(i, j.cand)]);
                let wt = K / (K + (sum as f64) / ((self.ranks - 1) as f64));
                for j in self.remaining.iter() {
                    self.wtd_scores[*j] += wt * (self.ballots[(i, *j)] as f64);
                }
            }
            //println!("self.wtd_scores = {:?}", self.wtd_scores);
            // let winner = remaining.iter()
            //                       .max_by_key(|&j| self.wtd_scores[*j]).unwrap();
            // let winner_idx = remaining.iter().find(|&j| j == winner).unwrap();
            let (winner_idx, winner_score) = {
                let mut rem_iter = self.remaining.iter();
                let mut winner_idx = 0;
                let mut winner_score = self.wtd_scores[*rem_iter.next().unwrap()];
                //println!("     winner = {}, score = {}", remaining[winner_idx], winner_score);
                for (idx, j) in rem_iter.enumerate() {
                    if self.wtd_scores[*j] > winner_score {
                        winner_idx = idx + 1;
                        winner_score = self.wtd_scores[*j];
                        //println!("     New winner={}, score={}", remaining[winner_idx], winner_score);
                    }
                }
                (winner_idx, winner_score)
            };
            let winner = self.remaining.swap_remove(winner_idx);
            self.winners.push(ElectResult{cand: winner, score: winner_score});
        }
        &self.winners
    }
}
