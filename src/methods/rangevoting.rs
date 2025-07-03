use ndarray::{ArrayView, Ix1};
use serde::{Deserialize, Serialize};

use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use super::MethodSim;
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RangeVoting {
    pub strat: Strategy,
    pub nranks: i32,
    #[serde(default = "default_stretch")]
    strategic_stretch_factor: f64,
}

fn default_stretch() -> f64 {
    4.0
}

#[derive(Debug)]
pub struct RangeVotingSim {
    params: RangeVoting,
    tallies: Tallies,
    ballot: Tallies,
}

impl RangeVoting {
    pub fn new_sim(&self, sim: &Sim) -> RangeVotingSim {
        RangeVotingSim {
            params: self.clone(),
            tallies: vec![0; sim.ncand],
            ballot: vec![0; sim.ncand],
        }
    }
}

impl MethodSim for RangeVotingSim {
    fn elect(&mut self, sim: &Sim, honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        self.tallies.fill(0);
        // for icit in 0..sim.ncit {
        for vscores in sim.scores.outer_iter() {
            match self.params.strat {
                Strategy::Honest => {
                    fill_range_ballot(&vscores, self.params.nranks, &mut self.ballot);
                }
                Strategy::Strategic => {
                    let pre_election = honest_rslt.unwrap();
                    let score_break = (vscores[pre_election.winner.cand]
                        + vscores[pre_election.runnerup.cand])
                        / 2.0;
                    fill_range_ballot_strat(
                        &vscores,
                        self.params.nranks,
                        &mut self.ballot,
                        score_break,
                        self.params.strategic_stretch_factor,
                    );
                }
            }
            for icand in 0..vscores.len() {
                self.tallies[icand] += self.ballot[icand];
            }
        }
        log::debug!("{} tallies: {:?}", self.name(), self.tallies);
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        if self.params.nranks == 2 {
            format!("Approval, {:?}", self.params.strat)
        } else {
            format!("Range 1-{}, {:?}", self.params.nranks, self.params.strat)
        }
    }

    fn colname(&self) -> String {
        if self.params.nranks == 2 {
            match self.params.strat {
                Strategy::Honest => format!("aprv_h"),
                Strategy::Strategic => format!("aprv_s"),
            }
        } else {
            match self.params.strat {
                Strategy::Honest => format!("range_{}_h", self.params.nranks),
                Strategy::Strategic => format!("range_{}_s", self.params.nranks),
            }
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}

pub fn fill_range_ballot(scores: &ArrayView<f64, Ix1>, ranks: i32, ballot: &mut [i32]) {
    let min_score = scores.iter().map(|x| *x).reduce(f64::min).unwrap();
    let max_score = scores.iter().map(|x| *x).reduce(f64::max).unwrap();
    // "ranks - 1" -- we give half a ranksz to the max and min score regions,
    // making our max and min score more likely to be given to only one candidate.
    let ranksz = (max_score - min_score) / ((ranks - 1) as f64);
    for (icand, score) in scores.iter().enumerate() {
        // We use a half rank size for scores of 0 and nranks-1.
        let r = ((score - min_score) / ranksz + 0.5).floor() as i32;
        ballot[icand] = r;
    }
}

pub fn fill_range_ballot_strat(
    scores: &ArrayView<f64, Ix1>,
    ranks: i32,
    ballot: &mut [i32],
    score_break: f64,
    stretch_factor: f64,
) {
    let min_score = scores.iter().map(|x| *x).reduce(f64::min).unwrap();
    let max_score = scores.iter().map(|x| *x).reduce(f64::max).unwrap();
    let score_range = max_score - min_score;
    let stretched_max = min_score + score_range * stretch_factor;
    let ranksz = score_range * stretch_factor / ((ranks - 1) as f64);
    for (icand, &score) in scores.iter().enumerate() {
        let mod_score = if score < score_break {
            score
        } else {
            stretched_max - (max_score - score)
        };
        let r = ((mod_score - min_score) / ranksz + 0.5).floor() as i32;
        ballot[icand] = r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    #[test]
    fn test_fill_range() {
        let scores = ndarray::Array::from_iter((0..100).map(|i| -9.111 + i as f64));
        let mut ballot = vec![0; 100];
        fill_range_ballot(&scores.view(), 100, &mut ballot);
        for (i, &b) in ballot.iter().enumerate() {
            assert_eq!(i as i32, b);
        }

        let scores = ndarray::array![10., 10.9, 12.7, 18.1, 19.0];
        // With a range of 0-9, 10 equal steps are 0.9 each.
        let mut ballot = vec![999; 5]; // fill value gets overwritten (prove this)
        fill_range_ballot_strat(&scores.view(), 100, &mut ballot, 11.5, 10.0);
        assert_eq!(ballot, &[0, 1, 92, 98, 99]);
    }

    #[test]
    fn test_range_voting() {
        let mut sim = Sim::new(3, 6);
        sim.scores = ndarray::array![
            [0., 5., 9.], // with strategy: 0., 9., 9.
            [0., 5., 9.], //                0., 9., 9.
            [1., 0., 9.], //                0., 0., 9.
            [9., 0., 2.], //                9., 0., 0.
            [9., 5., 0.], //                9., 9., 0.
            [9., 5., 0.], //                9., 9., 0.
                          // ttls: 28, 20, 29            27, 36, 27
        ];
        let mut method = RangeVoting {
            strat: Strategy::Honest,
            nranks: 10,
            strategic_stretch_factor: 2.0,
        }
        .new_sim(&sim);
        let honest_results = method.elect(&sim, None);
        assert_eq!(honest_results.winner.cand, 2);
        assert_eq!(honest_results.runnerup.cand, 0);

        let mut method2 = RangeVoting {
            strat: Strategy::Strategic,
            nranks: 10,
            strategic_stretch_factor: 1000., // votes become 0's and 10's
        }
        .new_sim(&sim);
        let strat_results = method2.elect(&sim, Some(honest_results));
        assert_eq!(strat_results.winner.cand, 1);
    }
}
