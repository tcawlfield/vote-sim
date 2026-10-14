// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! Proportional Approval Voting: the committee that maximizes the voters'
//! total harmonic satisfaction, found by searching the committees.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::ColName;
use super::MWMethodSim;
use super::rangevoting::fill_range_ballot;
use super::results::ElectResult;
use crate::committee_welfare::OwaSearch;
use crate::sim::Sim;

/// Proportional Approval Voting (PAV): each voter is satisfied 1 by their
/// favorite member of a committee they approve of, 1/2 by the next, 1/3 by the
/// next, and so on; the committee with the most satisfaction in total wins.
///
/// With `ranks` above 2 (the default), ballots are scores rather than
/// approvals: a member scored `s` out of `ranks - 1` counts for `s / (ranks -
/// 1)` of an approval, members sorted best first. RRV with `k = 1` is the
/// sequential (greedy) version of approval PAV.
///
/// ```toml
/// [[committee_methods]]
/// PAV = {}                # column pav_2
/// [[committee_methods]]
/// PAV = { ranks = 10 }    # column pav_10
/// ```
///
/// The best committee is found exactly, by a branch-and-bound search over all
/// of them (the same search as for Harmonic committee welfare); that can be
/// slow for a large field. Of committees that tie, the first found wins.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PAV {
    /// Scores on a ballot, from 0 to `ranks - 1`: 2 is approval voting.
    #[serde(default = "default_ranks")]
    pub ranks: i32,
    /// Replaces the default output column name; see [`ColName`].
    #[serde(default, skip_serializing_if = "ColName::is_default")]
    pub colname: ColName,
}

fn default_ranks() -> i32 {
    2
}

pub struct PAVSim {
    p: PAV,
    /// Every voter's ballot, scaled to [0, 1]: `ncand` x `nvtr`, so that one
    /// candidate's scores are contiguous.
    ballots_t: Array2<f64>,
    /// One voter's ballot, in ranks.
    ballot: Vec<i32>,
    /// The search, sized for the committee on the first trial.
    search: Option<OwaSearch>,
    winners: Vec<ElectResult>,
}

impl PAV {
    /// Check the parameters: at least two scores per ballot.
    pub fn validate(&self) -> Result<(), String> {
        if self.ranks < 2 {
            return Err(format!("PAV needs ranks >= 2, not {}", self.ranks));
        }
        Ok(())
    }

    /// The output column name for this method's results: the configured
    /// `colname`, or else the method's default.
    pub fn colname(&self) -> String {
        self.colname.or_else(|| format!("pav_{}", self.ranks))
    }

    pub fn new_sim(&self, sim: &Sim) -> PAVSim {
        PAVSim {
            p: self.clone(),
            ballots_t: Array2::zeros((sim.ncand, sim.nvtr)),
            ballot: vec![0; sim.ncand],
            search: None,
            winners: Vec::with_capacity(sim.ncand),
        }
    }
}

impl PAVSim {
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

impl MWMethodSim for PAVSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult> {
        self.cast_ballots(sim);
        let search = match &mut self.search {
            Some(search) if search.committee_size() == nwinners => search,
            search => {
                // self.search is None or the committee size has changed, so make a new search.
                let harmonic: Vec<f64> = (1..=nwinners).map(|j| 1.0 / j as f64).collect();
                search.insert(OwaSearch::new(&harmonic, sim.ncand, sim.nvtr))
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
        if self.p.ranks == 2 {
            "PAV".to_string()
        } else {
            format!("PAV, scores 0-{}", self.p.ranks - 1)
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

    /// PAV's own measure of a committee, worked out directly from the ballots:
    /// each voter's scores for the members, best first, weighted 1, 1/2, ...
    fn pav_score(ballots_t: &Array2<f64>, committee: &[usize]) -> f64 {
        (0..ballots_t.ncols())
            .map(|ivtr| {
                let mut scores: Vec<f64> =
                    committee.iter().map(|&c| ballots_t[(c, ivtr)]).collect();
                scores.sort_by(|a, b| b.total_cmp(a));
                scores
                    .iter()
                    .enumerate()
                    .map(|(j, s)| s / (j + 1) as f64)
                    .sum::<f64>()
            })
            .sum()
    }

    fn pav(ranks: i32) -> PAV {
        PAV {
            ranks,
            colname: ColName::default(),
        }
    }

    /// PAV's committee scores as well as the best of all of them, by its own
    /// measure, for approval and score ballots and every committee size.
    #[test]
    fn pav_elects_a_best_committee() {
        random_elections(|sim| {
            for ranks in [2, 5] {
                for k in 1..sim.ncand {
                    let mut method = pav(ranks).new_sim(sim);
                    let committee: Vec<usize> =
                        method.multi_elect(sim, k).iter().map(|w| w.cand).collect();
                    let best = combinations(sim.ncand, k)
                        .iter()
                        .map(|c| pav_score(&method.ballots_t, c))
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut distinct = committee.clone();
                    distinct.sort_unstable();
                    distinct.dedup();
                    assert_eq!(distinct.len(), k);
                    approx::assert_abs_diff_eq!(
                        pav_score(&method.ballots_t, &committee),
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
        let mut committee: Vec<usize> = pav(2)
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
                let mut method = pav(2).new_sim(sim);
                let committee: Vec<usize> =
                    method.multi_elect(sim, k).iter().map(|w| w.cand).collect();
                let greedy: Vec<usize> = rrv
                    .new_sim(sim)
                    .multi_elect(sim, k)
                    .iter()
                    .map(|w| w.cand)
                    .collect();
                assert!(
                    pav_score(&method.ballots_t, &committee)
                        >= pav_score(&method.ballots_t, &greedy) - 1e-9
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
            let mut method = pav(2).new_sim(sim);
            assert_eq!(method.multi_elect(sim, 2).len(), 2);
            assert_eq!(method.multi_elect(sim, 1).len(), 1);
        });
    }

    #[test]
    fn colname_and_validation() {
        assert_eq!(pav(2).colname(), "pav_2");
        assert_eq!(pav(10).colname(), "pav_10");
        assert_eq!(
            pav(1).validate().unwrap_err(),
            "PAV needs ranks >= 2, not 1"
        );
    }
}
