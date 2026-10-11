// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use super::results::{ElectResult, Standings};

pub type Tallies = Vec<i32>;

/// Fill `standings` from vote tallies: most votes first, ties in candidate
/// order -- except that a tie for first is settled by a coin toss.
pub fn tally_votes(tallies: &Tallies, standings: &mut Standings) {
    standings.clear();
    for (icand, &tally) in tallies.iter().enumerate() {
        standings.push(icand, tally as f64);
    }
    standings.order_mut().sort_unstable_by(most_first);
    if standings.is_tied() && rand::random() {
        // 50/50 chance
        standings.swap_top_two();
    }
}

/// Fill `standings` at the end of an elimination count: the `winner`, the
/// `runnerup` when the count named one, the other candidates still standing by
/// their final votes (ties in candidate order), then the eliminated, last out
/// first, each with their votes in the round they went out.
pub fn standings_after_elimination(
    standings: &mut Standings,
    winner: ElectResult,
    runnerup: Option<ElectResult>,
    tallies: &Tallies,
    eliminated: &[bool],
    out_order: &[ElectResult],
) {
    standings.clear();
    standings.push(winner.cand, winner.score);
    if let Some(r) = runnerup {
        standings.push(r.cand, r.score);
    }
    let placed = standings.order().len();
    for (icand, &tally) in tallies.iter().enumerate() {
        let is_placed = icand == winner.cand || runnerup.is_some_and(|r| r.cand == icand);
        if !eliminated[icand] && !is_placed {
            standings.push(icand, tally as f64);
        }
    }
    standings.order_mut()[placed..].sort_unstable_by(most_first);
    for out in out_order.iter().rev() {
        standings.push(out.cand, out.score);
    }
}

/// Highest score first, ties in candidate order.
fn most_first(a: &ElectResult, b: &ElectResult) -> Ordering {
    b.score.total_cmp(&a.score).then(a.cand.cmp(&b.cand))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(standings: &Standings) -> Vec<usize> {
        standings.order().iter().map(|r| r.cand).collect()
    }

    #[test]
    fn tally_votes_orders_by_votes_then_candidate() {
        let mut standings = Standings::default();
        tally_votes(&vec![3, 7, 3, 1, 5], &mut standings);
        assert_eq!(order(&standings), [1, 4, 0, 2, 3]);
        assert_eq!(standings.winner().score, 7.0);
    }

    fn result(cand: usize, score: f64) -> ElectResult {
        ElectResult { cand, score }
    }

    /// Winner, runner-up, the rest still standing by votes, then the
    /// eliminated, last out first.
    #[test]
    fn standings_after_elimination_puts_the_eliminated_last() {
        let tallies = vec![4, 0, 6, 1, 0];
        let eliminated = [false, true, false, false, true];
        let out_order = [result(4, 1.0), result(1, 2.0)];
        let mut standings = Standings::default();

        standings_after_elimination(
            &mut standings,
            result(2, 6.0),
            Some(result(0, 4.0)),
            &tallies,
            &eliminated,
            &out_order,
        );
        assert_eq!(order(&standings), [2, 0, 3, 1, 4]);
        assert_eq!(standings.order()[3].score, 2.0); // its votes when it went out

        // Without a runner-up the next best still standing comes second.
        standings_after_elimination(
            &mut standings,
            result(2, 6.0),
            None,
            &tallies,
            &eliminated,
            &out_order,
        );
        assert_eq!(order(&standings), [2, 0, 3, 1, 4]);
    }
}
