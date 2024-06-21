use super::method::{ElectResult, WinnerAndRunnerup};

pub type Tallies = Vec<i32>;

pub fn tally_votes(tallies: &Tallies) -> WinnerAndRunnerup {
    let ncand = tallies.len();
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
    for j in 2..ncand {
        if tallies[j] > most_votes {
            runup = electee;
            runup_votes = most_votes;
            electee = j;
            most_votes = tallies[j];
        } else if tallies[j] > runup_votes {
            runup = j;
            runup_votes = tallies[j];
        }
    }
    if most_votes == runup_votes {
        if rand::random() {
            // 50/50 chance
            (electee, runup) = (runup, electee);
            (most_votes, runup_votes) = (runup_votes, most_votes);
        }
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
