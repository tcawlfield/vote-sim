// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! Harmonic voting, which includes Proportional Approval Voting: the committee
//! that maximizes the voters' total harmonically weighted satisfaction, found
//! by searching the committees.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::ColName;
use super::MWMethodSim;
use super::rangevoting::fill_range_ballot;
use super::results::ElectResult;
use crate::committee_welfare::OwaSearch;
use crate::sim::Sim;

/// Warren Smith's harmonic voting
/// (<https://rangevoting.org/QualityMulti.html>): the committee with the most
/// voter satisfaction in total, where each voter's scores for its members,
/// best first, count `1/k`, `1/(1 + k)`, `1/(2 + k)`, and so on. Scores run
/// from 0 to `ranks - 1`, scaled to [0, 1].
///
/// With approval ballots (`ranks = 2`) and `k = 1`, both the defaults, this is
/// Proportional Approval Voting (PAV), whose weights are 1, 1/2, 1/3, ...;
/// its column is then `pav`, and otherwise `harm_<ranks>`. `k` is Smith's
/// delta: `k = 1` corresponds to D'Hondt apportionment and `k = 1/2` to
/// Sainte-Lague, which he conjectures is best. RRV with the same `ranks` and
/// `k` is a sequential (greedy) relative, the same method exactly for
/// approval ballots and `k = 1`.
///
/// ```toml
/// [[committee_methods]]
/// Harmonic = {}                       # PAV, column pav
/// [[committee_methods]]
/// Harmonic = { ranks = 10, k = 0.5 }  # column harm_10
/// ```
///
/// As for RRV, `k` isn't part of the column name; give each a `colname` to
/// compare several `k`.
///
/// The best committee is found exactly, by a branch-and-bound search over all
/// of them (the same search as for Harmonic committee welfare); that can be
/// slow for a large field. Of committees that tie, the first found wins.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Harmonic {
    /// Scores on a ballot, from 0 to `ranks - 1`: 2 is approval voting.
    #[serde(default = "default_ranks")]
    pub ranks: i32,
    /// Smith's delta: the `j`-th best member of a voter's committee counts
    /// `1/(j - 1 + k)`. From 1/2 to 1.
    #[serde(default = "default_k")]
    pub k: f64,
    /// Replaces the default output column name; see [`ColName`].
    #[serde(default, skip_serializing_if = "ColName::is_default")]
    pub colname: ColName,
}

fn default_ranks() -> i32 {
    2
}

fn default_k() -> f64 {
    1.0
}

pub struct HarmonicSim {
    p: Harmonic,
    /// Every voter's ballot, scaled to [0, 1]: `ncand` x `nvtr`, so that one
    /// candidate's scores are contiguous.
    ballots_t: Array2<f64>,
    /// One voter's ballot, in ranks.
    ballot: Vec<i32>,
    /// The search, sized for the committee on the first trial.
    search: Option<OwaSearch>,
    winners: Vec<ElectResult>,
}

impl Harmonic {
    /// Check the parameters: at least two scores per ballot, and `k` in the
    /// range Smith recommends (as for RRV).
    pub fn validate(&self) -> Result<(), String> {
        if self.ranks < 2 {
            return Err(format!("Harmonic needs ranks >= 2, not {}", self.ranks));
        }
        if !(0.5..=1.0).contains(&self.k) {
            return Err(format!("Harmonic needs 0.5 <= k <= 1.0, not {}", self.k));
        }
        Ok(())
    }

    /// Whether this is Proportional Approval Voting: approval ballots, `k = 1`.
    fn is_pav(&self) -> bool {
        self.ranks == 2 && self.k == 1.0
    }

    /// The output column name for this method's results: the configured
    /// `colname`, or else the method's default.
    pub fn colname(&self) -> String {
        self.colname.or_else(|| {
            if self.is_pav() {
                "pav".to_string()
            } else {
                format!("harm_{}", self.ranks)
            }
        })
    }

    /// Each member's weight, best first, in a committee of `nwinners`.
    fn weights(&self, nwinners: usize) -> Vec<f64> {
        (0..nwinners).map(|j| 1.0 / (j as f64 + self.k)).collect()
    }

    pub fn new_sim(&self, sim: &Sim) -> HarmonicSim {
        HarmonicSim {
            p: self.clone(),
            ballots_t: Array2::zeros((sim.ncand, sim.nvtr)),
            ballot: vec![0; sim.ncand],
            search: None,
            winners: Vec::with_capacity(sim.ncand),
        }
    }
}

impl HarmonicSim {
    /// Fill in every voter's ballot, scaled to [0, 1].
    fn cast_ballots(&mut self, sim: &Sim) {
        let top = (self.p.ranks - 1) as f64;
        for (ivtr, scores) in sim.scores.outer_iter().enumerate() {
            fill_range_ballot(&scores, self.p.ranks, &mut self.ballot);
            for (icand, &rank) in self.ballot.iter().enumerate() {
                self.ballots_t[(icand, ivtr)] = rank as f64 / top;
            }
        }
    }
}

impl MWMethodSim for HarmonicSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult> {
        self.cast_ballots(sim);
        let search = match &mut self.search {
            Some(search) if search.committee_size() == nwinners => search,
            search => {
                // self.search is None or the committee size has changed, so make a new search.
                let weights = self.p.weights(nwinners);
                search.insert(OwaSearch::new(&weights, sim.ncand, sim.nvtr))
            }
        };
        search.best(self.ballots_t.view());
        self.winners.clear();
        for &icand in search.committee() {
            self.winners.push(ElectResult {
                cand: icand,
                score: self.ballots_t.row(icand).sum(),
            });
        }
        &self.winners
    }

    fn name(&self) -> String {
        if self.p.is_pav() {
            "PAV".to_string()
        } else {
            format!("Harmonic, scores 0-{}, k={}", self.p.ranks - 1, self.p.k)
        }
    }

    fn colname(&self) -> String {
        self.p.colname()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::test_utils::{random_elections, sim_from_scores};
    use crate::methods::{MultiWinMethod, RRV};

    /// Every k-subset of 0..m.
    fn combinations(m: usize, k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        (k - 1..m)
            .flat_map(|last| {
                combinations(last, k - 1).into_iter().map(move |mut c| {
                    c.push(last);
                    c
                })
            })
            .collect()
    }

    /// The method's own measure of a committee, worked out directly from the
    /// ballots: each voter's scores for the members, best first, weighted
    /// 1/k, 1/(1 + k), ...
    fn harmonic_score(ballots_t: &Array2<f64>, k: f64, committee: &[usize]) -> f64 {
        (0..ballots_t.ncols())
            .map(|ivtr| {
                let mut scores: Vec<f64> =
                    committee.iter().map(|&c| ballots_t[(c, ivtr)]).collect();
                scores.sort_by(|a, b| b.total_cmp(a));
                scores
                    .iter()
                    .enumerate()
                    .map(|(j, s)| s / (j as f64 + k))
                    .sum::<f64>()
            })
            .sum()
    }

    fn harmonic(ranks: i32, k: f64) -> Harmonic {
        Harmonic {
            ranks,
            k,
            colname: ColName::default(),
        }
    }

    fn pav() -> Harmonic {
        harmonic(2, 1.0)
    }

    /// The committee scores as well as the best of all of them, by the
    /// method's own measure, for approval and score ballots, both ends of the
    /// range of `k`, and every committee size.
    #[test]
    fn harmonic_elects_a_best_committee() {
        random_elections(|sim| {
            for (ranks, k) in [(2, 1.0), (2, 0.5), (5, 1.0), (5, 0.5)] {
                for nseats in 1..sim.ncand {
                    let mut method = harmonic(ranks, k).new_sim(sim);
                    let committee: Vec<usize> = method
                        .multi_elect(sim, nseats)
                        .iter()
                        .map(|w| w.cand)
                        .collect();
                    let best = combinations(sim.ncand, nseats)
                        .iter()
                        .map(|c| harmonic_score(&method.ballots_t, k, c))
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut distinct = committee.clone();
                    distinct.sort_unstable();
                    distinct.dedup();
                    assert_eq!(distinct.len(), nseats);
                    approx::assert_abs_diff_eq!(
                        harmonic_score(&method.ballots_t, k, &committee),
                        best,
                        epsilon = 1e-9
                    );
                }
            }
        });
    }

    /// The textbook case: 6 voters approve A1-A3, 3 approve B1-B3. Three seats
    /// split 2 to 1 (satisfaction 6 * 3/2 + 3 = 12) rather than going all to
    /// the A's (6 * 11/6 = 11).
    #[test]
    fn pav_is_proportional() {
        let mut sim = sim_from_scores(&[
            (&[1., 1., 1., 0., 0., 0.], 6),
            (&[0., 0., 0., 1., 1., 1.], 3),
        ]);
        sim.rank_candidates();
        let mut committee: Vec<usize> = pav()
            .new_sim(&sim)
            .multi_elect(&sim, 3)
            .iter()
            .map(|w| w.cand)
            .collect();
        committee.sort_unstable();
        assert_eq!(
            committee.iter().filter(|&&c| c < 3).count(),
            2,
            "{committee:?}"
        );
        assert_eq!(
            committee.iter().filter(|&&c| c >= 3).count(),
            1,
            "{committee:?}"
        );
    }

    /// RRV with k = 1 on approval ballots is sequential PAV, which can do no
    /// better than PAV by PAV's own measure.
    #[test]
    fn pav_does_at_least_as_well_as_its_sequential_version() {
        let rrv = MultiWinMethod::RRV(RRV {
            strat: crate::methods::Strategy::Honest,
            ranks: 2,
            k: 1.0,
            colname: ColName::default(),
        });
        random_elections(|sim| {
            for k in 1..sim.ncand {
                let mut method = pav().new_sim(sim);
                let committee: Vec<usize> =
                    method.multi_elect(sim, k).iter().map(|w| w.cand).collect();
                let greedy: Vec<usize> = rrv
                    .new_sim(sim)
                    .multi_elect(sim, k)
                    .iter()
                    .map(|w| w.cand)
                    .collect();
                assert!(
                    harmonic_score(&method.ballots_t, 1.0, &committee)
                        >= harmonic_score(&method.ballots_t, 1.0, &greedy) - 1e-9
                );
            }
        });
    }

    /// The search is reused while the committee size stays the same, and
    /// replaced when it changes.
    #[test]
    fn a_new_committee_size_gets_a_new_search() {
        random_elections(|sim| {
            if sim.ncand < 3 {
                return;
            }
            let mut method = pav().new_sim(sim);
            assert_eq!(method.multi_elect(sim, 2).len(), 2);
            assert_eq!(method.multi_elect(sim, 1).len(), 1);
        });
    }

    /// The column is `pav` exactly when the method is PAV.
    #[test]
    fn colname_and_validation() {
        assert_eq!(pav().colname(), "pav");
        assert_eq!(harmonic(2, 0.5).colname(), "harm_2");
        assert_eq!(harmonic(10, 1.0).colname(), "harm_10");
        assert_eq!(
            harmonic(1, 1.0).validate().unwrap_err(),
            "Harmonic needs ranks >= 2, not 1"
        );
        assert_eq!(
            harmonic(2, 0.4).validate().unwrap_err(),
            "Harmonic needs 0.5 <= k <= 1.0, not 0.4"
        );
    }
}
