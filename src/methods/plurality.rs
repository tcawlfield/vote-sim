// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use super::MethodSim;
use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{Tallies, tally_votes};
use crate::sim::Sim;
use serde::{Deserialize, Serialize};

/// In Plurality voting, a.k.a First Past the Post, voters mark exactly one candidate on
/// their ballots to indicate that candidate as their top preference. The candidate
/// with the must number of "votes" wins the election.
///
/// Although this method is very simple and intuitive, and by far the most widely used,
/// it performs much worse than all other methods in almost every way, regardless of
/// the details of the candidates-voter consideration model.
///
/// The poor performance of Plurality voting is counterintuitive, but there are at least
/// three major problems identified with it, which can help to understand it more
/// intuitively.
///
/// 1. Very little information is expressed on the ballot. A voter is unable to communicate
///    anything beyond their preference for one candidate.
/// 2. There is a strong spoiler effect. If a candidate enters an election who is
///    relatively unpopular, they "steal" a small number of votes from the popular candidates.
///    Most of these votes are from voters that would otherwise prefer the popular
///    candidate nearest the "spoiler" candidate, and thus lower the chance of that
///    popular candidate winning. This tends to drive a political system into a
///    two-party equilibrium, as decribed by Duverger's law.
/// 3. Related to (2), Plurality has a "center-squeeze" effect, that supresses the
///    ability for centrist candidates to win an election. This effect increases
///    with the number of candidates, which makes open primary elections likely to
///    favor outlying candidates, or extremists.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plurality {
    /// Honest voters will vote for the candidate with the highest score, or
    /// perceived utility. Strategic voters will instead limit their choice to
    /// one of the two front-runners in a pre-election poll.
    pub strat: Strategy,
}

#[derive(Debug)]
pub struct PluralitySim {
    params: Plurality,
    tallies: Tallies,
}

impl Plurality {
    pub fn new_sim(&self, sim: &Sim) -> PluralitySim {
        PluralitySim {
            params: self.clone(),
            tallies: vec![0; sim.ncand],
        }
    }
}

impl MethodSim for PluralitySim {
    fn elect(&mut self, sim: &Sim, honest_rslt: Option<WinnerAndRunnerup>) -> WinnerAndRunnerup {
        match self.params.strat {
            Strategy::Honest => {
                self.tallies.fill(0);
                for &ivtr in sim.ranks.column(0) {
                    self.tallies[ivtr] += 1;
                }
            }
            Strategy::Strategic => {
                let pre_poll = if let Some(prev) = honest_rslt {
                    prev
                } else {
                    self.params.strat = Strategy::Honest;
                    let prev = self.elect(sim, None);
                    self.params.strat = Strategy::Strategic;
                    prev
                };
                self.tallies.fill(0);
                for vtr_ranks in sim.ranks.rows() {
                    for &icand in vtr_ranks {
                        if icand == pre_poll.winner.cand || icand == pre_poll.runnerup.cand {
                            self.tallies[icand] += 1;
                            break;
                        }
                    }
                }
            }
        }
        log::debug!(
            "Plurality votes ({:?}): {:?}",
            self.params.strat,
            self.tallies
        );
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        format!("Plurality, {:?}", self.params.strat)
    }

    fn colname(&self) -> String {
        match self.params.strat {
            Strategy::Honest => "pl_h".to_string(),
            Strategy::Strategic => "pl_s".to_string(),
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::ElectResult;
    use crate::methods::test_utils::sim_from_scores;
    use crate::sim::Sim;

    fn honest(sim: &Sim) -> PluralitySim {
        Plurality {
            strat: Strategy::Honest,
        }
        .new_sim(sim)
    }

    fn strategic(sim: &Sim) -> PluralitySim {
        Plurality {
            strat: Strategy::Strategic,
        }
        .new_sim(sim)
    }

    #[test]
    fn honest_plurality_counts_first_choices() {
        let mut sim = sim_from_scores(&[
            (&[3., 1., 0.], 5), // 5 voters rank candidate 0 first
            (&[1., 3., 0.], 3), // 3 rank candidate 1 first
            (&[0., 1., 3.], 2), // 2 rank candidate 2 first
        ]);
        sim.rank_candidates();

        let mut method = honest(&sim);
        let result = method.elect(&sim, None);

        assert_eq!(method.tallies, vec![5, 3, 2]);
        assert_eq!(result.winner.cand, 0);
        assert_eq!(result.winner.score, 5.0);
        assert_eq!(result.runnerup.cand, 1);
        assert_eq!(result.runnerup.score, 3.0);
    }

    #[test]
    fn honest_plurality_ignores_honest_rslt_argument() {
        let mut sim =
            sim_from_scores(&[(&[3., 1., 0.], 5), (&[1., 3., 0.], 3), (&[0., 1., 3.], 2)]);
        sim.rank_candidates();

        let bogus = WinnerAndRunnerup {
            winner: ElectResult {
                cand: 2,
                score: 99.0,
            },
            runnerup: ElectResult {
                cand: 2,
                score: 99.0,
            },
        };
        let with_hint = honest(&sim).elect(&sim, Some(bogus));
        let without_hint = honest(&sim).elect(&sim, None);
        assert_eq!(with_hint.winner.cand, without_hint.winner.cand);
        assert_eq!(with_hint.winner.cand, 0);
    }

    #[test]
    fn honest_plurality_elects_a_polarizing_candidate_over_a_consensus_one() {
        // Candidate 2 is every voter's 2nd choice (a Condorcet winner) yet gets
        // no first-choice votes, so Plurality never considers it.
        let mut sim = sim_from_scores(&[
            (&[3., 0., 2.], 5), // 0 > 2 > 1
            (&[0., 3., 2.], 4), // 1 > 2 > 0
        ]);
        sim.rank_candidates();

        let mut method = honest(&sim);
        let result = method.elect(&sim, None);

        assert_eq!(method.tallies, vec![5, 4, 0]);
        assert_eq!(result.winner.cand, 0);
    }

    #[test]
    fn strategic_plurality_defects_to_the_preferred_frontrunner() {
        let mut sim = sim_from_scores(&[
            (&[3., 2., 0.], 4), // 0 > 1 > 2
            (&[0., 3., 1.], 3), // 1 > 2 > 0
            (&[0., 2., 3.], 2), // 2 > 1 > 0
        ]);
        sim.rank_candidates();

        let honest_result = honest(&sim).elect(&sim, None);
        assert_eq!(honest_result.winner.cand, 0);
        assert_eq!(honest_result.runnerup.cand, 1);

        let mut method = strategic(&sim);
        let result = method.elect(&sim, Some(honest_result));

        // Frontrunners are 0 and 1. The 2 candidate-2 voters prefer 1 to 0, so
        // they abandon 2 and vote 1, which overtakes 0.
        assert_eq!(method.tallies, vec![4, 5, 0]);
        assert_eq!(result.winner.cand, 1);
    }

    #[test]
    fn strategic_plurality_runs_its_own_pre_poll_when_none_is_given() {
        let mut sim =
            sim_from_scores(&[(&[3., 2., 0.], 4), (&[0., 3., 1.], 3), (&[0., 2., 3.], 2)]);
        sim.rank_candidates();

        let mut method = strategic(&sim);
        let result = method.elect(&sim, None);

        assert_eq!(method.tallies, vec![4, 5, 0]);
        assert_eq!(result.winner.cand, 1);
        // The temporary switch to Honest for the pre-poll is reverted.
        assert!(matches!(method.strat(), Strategy::Strategic));
    }

    #[test]
    fn method_metadata() {
        let sim = Sim::new(3, 1);

        let h = honest(&sim);
        assert_eq!(h.colname(), "pl_h");
        assert_eq!(h.name(), "Plurality, Honest");
        assert!(matches!(h.strat(), Strategy::Honest));

        let s = strategic(&sim);
        assert_eq!(s.colname(), "pl_s");
        assert_eq!(s.name(), "Plurality, Strategic");
        assert!(matches!(s.strat(), Strategy::Strategic));
    }
}
