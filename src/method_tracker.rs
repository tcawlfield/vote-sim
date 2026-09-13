// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use meansd::MeanSD;

use crate::methods::{MWMethodSim, Method, MethodSim, MultiWinMethod, WinnerAndRunnerup};
use crate::out_types::{CommitteeMethodResult, MethodResult};
use crate::sim::Sim;

/// Tracks one single-winner method across trials, recording the mean and standard deviation
/// of the regret of the elected winner, and the fraction of trials where the winner
/// was sub-optimal.
pub struct MethodTracker {
    pub method: Box<dyn MethodSim>,
    /// The number of trials run so far.
    ntrials: usize,
    /// The number of trials where the winner was not the one with highest utility.
    ntrials_subopt: usize,
    /// The mean and standard deviation of the regret of the winner.
    mean_regret: MeanSD,
    /// The mean and standard deviation of the regret of the winner
    /// on trials where the winner was sub-optimal.
    mean_subopt_regret: MeanSD,
}

impl MethodTracker {
    pub fn new(method: &Method, sim: &Sim) -> MethodTracker {
        MethodTracker {
            method: method.new_sim(sim),
            ntrials: 0,
            ntrials_subopt: 0,
            mean_regret: MeanSD::default(),
            mean_subopt_regret: MeanSD::default(),
        }
    }

    /// Run the method for one trial. Returns the winner/runner-up pairing (for
    /// the honest pre-poll that strategic methods consume) and the per-trial
    /// [`MethodResult`] destined for the output.
    pub fn elect(
        &mut self,
        sim: &Sim,
        honest_rslt: Option<WinnerAndRunnerup>,
    ) -> (WinnerAndRunnerup, MethodResult) {
        let mut result = self.method.elect(sim, honest_rslt);

        if result.is_tied() {
            result = sim.break_tie_with_plurality(&result);
        }

        self.ntrials += 1;
        let regret = sim.regrets[result.winner.cand];
        self.mean_regret.update(regret);
        if regret > 0.0 {
            self.ntrials_subopt += 1;
            self.mean_subopt_regret.update(regret);
        }

        let method_result = MethodResult {
            winner: sim.regret_rank[result.winner.cand] as u32,
            regret,
        };
        (result, method_result)
    }

    pub fn colname(&self) -> String {
        self.method.colname()
    }

    #[allow(dead_code)]
    pub fn report(&self) {
        let frac_suboptimal = self.ntrials_subopt as f64 / self.ntrials as f64;
        println!(
            "Method {}: Avg Regret: {}, σ: {}, Frac suboptimal winner: {}, avg subopt regret: {}",
            self.method.name(),
            self.mean_regret.mean(),
            self.mean_regret.sstdev(),
            frac_suboptimal,
            self.mean_subopt_regret.mean(),
        )
    }

    pub fn sendable_report(&self) -> SendableMethodReport {
        SendableMethodReport {
            name: self.method.name(),
            ntrials: self.ntrials,
            ntrials_subopt: self.ntrials_subopt,
            mean_regret: self.mean_regret,
            mean_subopt_regret: self.mean_subopt_regret,
        }
    }
}

/// Tracks one multi-winner ("committee") method across trials, the multi-winner
/// counterpart to [`MethodTracker`].
///
/// Regret here is the mean regret of the elected committee's members. It's a
/// simple utilitarian measure -- it does not reward a method for proportionally
/// representing a minority faction -- but it stays on the same scale as
/// single-winner regret and costs nothing extra to compute.
pub struct CommitteeTracker {
    pub method: Box<dyn MWMethodSim>,
    committee_size: usize,
    ntrials: usize,
    ntrials_subopt: usize,
    mean_regret: MeanSD,
    mean_subopt_regret: MeanSD,
}

impl CommitteeTracker {
    pub fn new(method: &MultiWinMethod, sim: &Sim, committee_size: usize) -> CommitteeTracker {
        CommitteeTracker {
            method: method.new_sim(sim),
            committee_size,
            ntrials: 0,
            ntrials_subopt: 0,
            mean_regret: MeanSD::default(),
            mean_subopt_regret: MeanSD::default(),
        }
    }

    /// Run the method for one trial, electing `committee_size` winners.
    pub fn elect(&mut self, sim: &Sim) -> CommitteeMethodResult {
        // Cloned (cheap: committee_size is small) so the tracker's own fields
        // can be updated afterward without fighting the borrow on self.method.
        let winners = self.method.multi_elect(sim, self.committee_size).clone();

        let regret =
            winners.iter().map(|w| sim.regrets[w.cand]).sum::<f64>() / winners.len() as f64;
        self.ntrials += 1;
        self.mean_regret.update(regret);
        if regret > 0.0 {
            self.ntrials_subopt += 1;
            self.mean_subopt_regret.update(regret);
        }

        CommitteeMethodResult {
            winners: winners
                .iter()
                .map(|w| sim.regret_rank[w.cand] as u32)
                .collect(),
            regret,
        }
    }

    pub fn colname(&self) -> String {
        self.method.colname()
    }

    #[allow(dead_code)]
    pub fn report(&self) {
        let frac_suboptimal = self.ntrials_subopt as f64 / self.ntrials as f64;
        println!(
            "Committee {}: Avg Regret: {}, σ: {}, Frac suboptimal winner: {}, avg subopt regret: {}",
            self.method.name(),
            self.mean_regret.mean(),
            self.mean_regret.sstdev(),
            frac_suboptimal,
            self.mean_subopt_regret.mean(),
        )
    }

    pub fn sendable_report(&self) -> SendableMethodReport {
        SendableMethodReport {
            name: self.method.name(),
            ntrials: self.ntrials,
            ntrials_subopt: self.ntrials_subopt,
            mean_regret: self.mean_regret,
            mean_subopt_regret: self.mean_subopt_regret,
        }
    }
}

/// The multi-winner counterpart to [`SendableMethodReport`].
pub struct SendableMethodReport {
    pub name: String,
    pub ntrials: usize,
    pub ntrials_subopt: usize,
    pub mean_regret: MeanSD,
    pub mean_subopt_regret: MeanSD,
}

impl SendableMethodReport {
    pub fn combine(&mut self, other: &Self) {
        assert!(self.name == other.name);
        self.ntrials += other.ntrials;
        self.ntrials_subopt += other.ntrials_subopt;
        self.mean_regret += other.mean_regret;
        self.mean_subopt_regret += other.mean_subopt_regret;
    }

    pub fn report(&self) {
        let frac_suboptimal = self.ntrials_subopt as f64 / self.ntrials as f64;
        println!(
            "Method {}: Avg Regret: {}, σ: {}, Frac suboptimal winner: {}, avg subopt regret: {}, {} elections",
            self.name,
            self.mean_regret.mean(),
            self.mean_regret.sstdev(),
            frac_suboptimal,
            self.mean_subopt_regret.mean(),
            self.ntrials,
        );
    }
}
