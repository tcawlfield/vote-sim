// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use super::results::{ElectResult, WinnerAndRunnerup};

pub type Tallies = Vec<i32>;

pub fn tally_votes(tallies: &Tallies) -> WinnerAndRunnerup {
    let mut electee = 0usize;
    let mut most_votes = tallies[0];
    let mut runup = 1usize;
    let mut runup_votes = tallies[1];
    if runup_votes > most_votes {
        electee = 1;
        runup = 0;
        most_votes = tallies[1];
        runup_votes = tallies[0];
    }
    for (j, &tally) in tallies.iter().enumerate().skip(2) {
        if tally > most_votes {
            runup = electee;
            runup_votes = most_votes;
            electee = j;
            most_votes = tally;
        } else if tally > runup_votes {
            runup = j;
            runup_votes = tally;
        }
    }
    if most_votes == runup_votes && rand::random() {
        // 50/50 chance
        (electee, runup) = (runup, electee);
        (most_votes, runup_votes) = (runup_votes, most_votes);
    }
    // TODO: We still are flubbing cases where there's a tie for runner-up or 3+-way for 1st.
    WinnerAndRunnerup {
        winner: ElectResult {
            cand: electee,
            score: most_votes as f64,
        },
        runnerup: ElectResult {
            cand: runup,
            score: runup_votes as f64,
        },
    }
}
