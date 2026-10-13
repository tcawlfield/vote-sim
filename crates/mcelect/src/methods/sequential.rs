// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! A committee method made from any single-winner one: run that election once
//! per seat, each time without the candidates already elected.

use serde::{Deserialize, Serialize};

use super::results::ElectResult;
use super::standalone::Standalone;
use super::{MWMethodSim, Method};
use crate::sim::Sim;

/// A committee elected one seat at a time by a single-winner method, each seat
/// a fresh election among the candidates not yet elected:
///
/// ```toml
/// [[committee_methods]]
/// Sequential = { InstantRunoff = {} }
/// ```
///
/// Its column is the single-winner method's, prefixed with `seq_` (`seq_IRV_h`
/// here). Unlike [`TopStandings`](super::TopStandings), a candidate who placed
/// low in one election -- squeezed out early by IRV, say -- is back in contention
/// for the next seat, so a majority can keep winning seats: this tends to be
/// the more majoritarian of the two. For Ranked Pairs, whose standings are
/// already found this way, the two are the same.
///
/// Voters cast fresh ballots for every seat, as if each were a separate vote
/// among the candidates left. For ranked ballots (IRV, BTR-IRV, Ranked Pairs,
/// Minimax, Borda) that's the same as recounting the original ballots with the
/// winners struck off. Plurality and score ballots differ: there a recount of
/// the original ballots would seat the same committee as `TopStandings`,
/// whereas fresh ballots move to the voter's next choice or are rescaled over
/// the smaller field. A strategic method runs its own honest poll for every
/// seat, so its voters re-strategize each time.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct Sequential(pub Method);

pub struct SequentialSim {
    method: Method,
    name: String,
    colname: String,
    /// One election per seat. Seat `s` chooses among `ncand - s` candidates.
    /// The committee size isn't known until the first trial, which allocates
    /// them; later trials reuse them.
    seats: Vec<Seat>,
    /// Whether each candidate of the full field has a seat yet.
    seated: Vec<bool>,
    /// The candidates still in the running for the current seat, by their
    /// index in the full field.
    remaining: Vec<ElectResult>,
    winners: Vec<ElectResult>,
}

/// The election for one seat.
struct Seat {
    /// Those still in the running for this seat. `None` for the first seat,
    /// which is the trial's whole field.
    field: Option<Sim>,
    election: Standalone,
}

impl Sequential {
    pub fn colname(&self) -> String {
        format!("seq_{}", self.0.colname())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate()
    }

    pub fn new_sim(&self, sim: &Sim) -> SequentialSim {
        SequentialSim {
            method: self.0.clone(),
            name: format!("Sequential {}", self.0.new_sim(sim).name()),
            colname: self.colname(),
            seats: Vec::with_capacity(sim.ncand),
            seated: vec![false; sim.ncand],
            remaining: Vec::with_capacity(sim.ncand),
            winners: Vec::with_capacity(sim.ncand),
        }
    }
}

impl SequentialSim {
    /// Make sure there's an election for each of `nseats` seats.
    fn allocate_seats(&mut self, sim: &Sim, nseats: usize) {
        while self.seats.len() < nseats {
            let iseat = self.seats.len();
            let field = Sim::new(sim.ncand - iseat, sim.nvtr);
            self.seats.push(Seat {
                election: Standalone::new(&self.method, &field),
                field: (iseat > 0).then_some(field),
            });
        }
    }
}

impl MWMethodSim for SequentialSim {
    fn multi_elect(&mut self, sim: &Sim, nwinners: usize) -> &Vec<ElectResult> {
        self.allocate_seats(sim, nwinners);
        self.seated.fill(false);
        self.winners.clear();
        for seat in self.seats[..nwinners].iter_mut() {
            // The candidates still in the running, in their original order.
            self.remaining.clear();
            self.remaining.extend(
                (0..sim.ncand)
                    .filter(|&icand| !self.seated[icand])
                    .map(|cand| ElectResult { cand, score: 0.0 }),
            );
            let field = match &mut seat.field {
                Some(field) => {
                    field.take_from_primary(sim, &self.remaining);
                    &*field
                }
                None => sim,
            };
            let winner = seat.election.elect(field).winner();
            // Back from this seat's field to the full one.
            let cand = self.remaining[winner.cand].cand;
            self.seated[cand] = true;
            self.winners.push(ElectResult {
                cand,
                score: winner.score,
            });
        }
        &self.winners
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn colname(&self) -> String {
        self.colname.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::TopStandings;
    use crate::methods::test_utils::{random_elections, sim_from_scores};
    use crate::methods::tests::every_method;

    fn seated(committee: &[ElectResult]) -> Vec<usize> {
        committee.iter().map(|w| w.cand).collect()
    }

    /// Every seat goes to a different candidate, and the first to the
    /// single-winner method's own winner.
    #[test]
    fn sequential_seats_distinct_candidates_starting_with_the_winner() {
        random_elections(|sim| {
            for method in every_method() {
                let winner = Standalone::new(&method, sim).elect(sim).winner().cand;
                for k in 1..sim.ncand {
                    // A fresh sim, so its coin for ties starts where `winner`'s did.
                    let mut seq = Sequential(method.clone()).new_sim(sim);
                    let mut committee = seated(seq.multi_elect(sim, k));
                    assert_eq!(committee[0], winner, "{method:?}, k = {k}");
                    committee.sort_unstable();
                    committee.dedup();
                    assert_eq!(committee.len(), k, "{method:?}, k = {k}");
                }
            }
        });
    }

    /// Ranked Pairs' standings are already found a seat at a time, so the two
    /// ways of making it a committee agree.
    #[test]
    fn sequential_ranked_pairs_matches_its_top_standings() {
        let rp: Method = serde_json::from_str(r#"{"RP": {"strat": "Honest"}}"#).unwrap();
        random_elections(|sim| {
            let mut seq = Sequential(rp.clone()).new_sim(sim);
            let mut top = TopStandings(rp.clone()).new_sim(sim);
            for k in 1..sim.ncand {
                assert_eq!(
                    seated(seq.multi_elect(sim, k)),
                    seated(top.multi_elect(sim, k)),
                    "k = {k}"
                );
            }
        });
    }

    /// The center squeeze. L and R voters each rank C second, but C is first
    /// choice for too few to survive IRV's first round, so C is IRV's last
    /// place and out of its top two. Run again without L, though, and L's
    /// voters carry C to the second seat.
    #[test]
    fn sequential_irv_gives_a_squeezed_candidate_another_chance() {
        let (l, c, r) = (0, 1, 2);
        let mut sim = sim_from_scores(&[
            (&[3., 2., 1.], 40), // L > C > R
            (&[1., 2., 3.], 35), // R > C > L
            (&[2., 3., 1.], 13), // C > L > R
            (&[1., 3., 2.], 12), // C > R > L
        ]);
        sim.compute_regrets();
        sim.rank_candidates();
        sim.find_smith_set();
        let irv: Method = serde_json::from_str(r#"{"InstantRunoff": {}}"#).unwrap();

        let top = seated(TopStandings(irv.clone()).new_sim(&sim).multi_elect(&sim, 2));
        assert_eq!(top, [l, r]); // C out first: 40 + 13 for L, 35 + 12 for R
        let seq = seated(Sequential(irv).new_sim(&sim).multi_elect(&sim, 2));
        assert_eq!(seq, [l, c]); // then C, 25 + 40, beats R, 35
    }

    #[test]
    fn colname_prefixes_the_methods() {
        let irv = Sequential(serde_json::from_str(r#"{"InstantRunoff": {}}"#).unwrap());
        assert_eq!(irv.colname(), "seq_IRV_h");
        assert_eq!(irv.new_sim(&Sim::new(3, 1)).colname(), "seq_IRV_h");
    }
}
