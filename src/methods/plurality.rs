use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use super::MethodSim;
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
                for icit in 0..sim.ncit {
                    self.tallies[sim.ranks[(icit, 0)]] += 1;
                }
            }
            Strategy::Strategic => {
                let pre_poll = if let Some(prev) = honest_rslt {
                    prev
                } else {
                    self.params.strat = Strategy::Honest;
                    let prev = self.elect(&sim, None);
                    self.params.strat = Strategy::Strategic;
                    prev
                };
                self.tallies.fill(0);
                for icit in 0..sim.ncit {
                    for rank in 0..sim.ncand {
                        let icand = sim.ranks[(icit, rank)];
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
            Strategy::Honest => format!("pl_h"),
            Strategy::Strategic => format!("pl_s"),
        }
    }

    fn strat(&self) -> Strategy {
        self.params.strat
    }
}
