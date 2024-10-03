use ndarray::Axis;
use serde::{Deserialize, Serialize};

use super::MethodSim;
use super::results::{Strategy, WinnerAndRunnerup};
use super::tallies::{tally_votes, Tallies};
use crate::sim::Sim;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Borda {
    // pub strat: Strategy,
}

#[derive(Debug)]
pub struct BordaSim {
    _p: Borda,
    tallies: Tallies,
}

impl Borda {
    pub fn new_sim(&self, sim: &Sim) -> BordaSim {
        BordaSim {
            _p: self.clone(),
            tallies: vec![0; sim.ncand],
        }
    }
}

impl MethodSim for BordaSim {
    fn elect(
        &mut self,
        sim: &Sim,
        _honest_rslt: Option<WinnerAndRunnerup>,
        verbose: bool,
    ) -> WinnerAndRunnerup {
        self.tallies.fill(0);
        for cand_fav_iter in sim.ranks.lanes(Axis(1)) {
            for (cand_rank, &icand) in cand_fav_iter.indexed_iter() {
                self.tallies[icand] += (sim.ncand - cand_rank) as i32;
            }
        }
        if verbose {
            println!("Borda tallies are: {:?}", self.tallies);
        }
        tally_votes(&self.tallies)
    }

    fn name(&self) -> String {
        format!("Borda, {}", "Honest")
    }

    fn colname(&self) -> String {
        "Borda_h".to_string()
        // match self.p.strat {
        //     Strategy::Honest => "IRV_h".to_string(),
        //     Strategy::Strategic => "IRV_s".to_string(),
        // }
    }

    fn strat(&self) -> Strategy {
        // self.p.strat
        Strategy::Honest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    #[test]
    fn test_borda_honest() {
        let mut sim = Sim::new(4, 5);
        sim.scores = ndarray::array![
            [4., 3., 2., 1.],
            [1., 4., 2., 3.],
            [3., 4., 2., 1.],
            [3., 2., 1., 4.],
            [3., 2., 4., 1.],
        ];
        // Tallies: 14, 15, 11, 10
        let mut method = Borda {}.new_sim(&sim);
        sim.rank_candidates();
        let honest_results = method.elect(&sim, None, true);
        assert_eq!(honest_results.winner.cand, 1);
        assert_eq!(honest_results.winner.score, 15.);
        assert_eq!(honest_results.runnerup.cand, 0);
        assert_eq!(honest_results.runnerup.score, 14.);
    }
}
