// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! One worker's state for a batch of trials, shared by both election modes.
//!
//! [`Runner::new`] allocates everything a batch needs once. Each call to
//! [`Runner::do_trial`] (single-winner) or [`Runner::do_committee_trial`]
//! (multi-winner) then runs one election and returns that trial's output row,
//! reusing all of it. Batch orchestration -- the worker pool, channels and
//! parquet writing -- lives in [`crate::run`] and [`crate::run_multi`].

use std::collections::BTreeMap;

use rand::Rng;

use crate::config::{Config, RunMode};
use crate::considerations::{ConsiderationSim, ConsiderationSimKind};
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::{CommitteeTracker, MethodTracker, SendableMethodReport};
use crate::methods::{MWMethodSim, Strategy};
use crate::out_types::{CommitteeMethodResult, CommitteeResult, ExperimentResult, MethodResult};
use crate::sim::Sim;

/// Everything one worker needs to run a batch of trials in either mode.
///
/// `methods` and `committees` are the only mode-specific members, and exactly
/// one of them is non-empty: the other mode's trackers are simply never built.
/// The primary-narrowing stage (`sim_primary` / `primary_method`) likewise
/// exists only in [`RunMode::SingleWinner`].
pub(crate) struct TrialRunner<R: Rng> {
    mode: RunMode,
    rng: R,
    sim: Sim,
    /// The larger pre-primary candidate field. `Some` only when the config has
    /// a primary stage, which is single-winner mode only.
    sim_primary: Option<Sim>,
    axes: Vec<ConsiderationSimKind>,
    /// Single-winner methods. Non-empty only in [`RunMode::SingleWinner`].
    methods: Vec<MethodTracker>,
    /// Multi-winner methods. Non-empty only in [`RunMode::MultiWinner`].
    committees: Vec<CommitteeTracker>,
    cov_matrix: CovMatrix,
    /// Sim state for the config's `primary_method`. `Some` exactly when
    /// `sim_primary` is.
    primary_method: Option<Box<dyn MWMethodSim>>,
    /// Candidates in order of increasing regret. With no primary this is
    /// identical to `sim.cand_by_regret`; with a primary it holds only the
    /// winning primary candidates. Kept here so its allocation is reused.
    ordered_final_cands: Vec<usize>,
    /// Trials run so far, for logging.
    itrial: usize,
}

impl<R: Rng> TrialRunner<R> {
    pub(crate) fn new(config: &Config, rng: R) -> TrialRunner<R> {
        let nvtr = config.voters;
        let sim = Sim::new(config.candidates, nvtr);

        // The primary-narrowing stage belongs to single-winner mode only;
        // committee methods always run against the full candidate field.
        let sim_primary = match config.mode {
            RunMode::SingleWinner => config.primary_candidates.map(|pcand| Sim::new(pcand, nvtr)),
            RunMode::MultiWinner => None,
        };

        let axes: Vec<ConsiderationSimKind> = {
            let max_sim = sim_primary.as_ref().unwrap_or(&sim);
            config
                .considerations
                .iter()
                .map(|c| c.new_sim(max_sim))
                .collect()
        };

        let (methods, committees): (Vec<MethodTracker>, Vec<CommitteeTracker>) = match config.mode {
            RunMode::SingleWinner => (
                config
                    .methods
                    .iter()
                    .map(|m| MethodTracker::new(m, &sim))
                    .collect(),
                Vec::new(),
            ),
            RunMode::MultiWinner => {
                let committee_size = config
                    .committee_size
                    .expect("Config::validate ensures committee_size is set in MultiWinner mode");
                (
                    Vec::new(),
                    config
                        .committee_methods
                        .iter()
                        .map(|m| CommitteeTracker::new(m, &sim, committee_size))
                        .collect(),
                )
            }
        };

        let cov_matrix = CovMatrix::new(sim.ncand);

        let primary_method = sim_primary
            .as_ref()
            .map(|sim_primary| config.primary_method.new_sim(sim_primary));

        let ordered_final_cands = vec![0; sim.ncand];

        TrialRunner {
            mode: config.mode,
            rng,
            sim,
            sim_primary,
            axes,
            methods,
            committees,
            cov_matrix,
            primary_method,
            ordered_final_cands,
            itrial: 0,
        }
    }

    /// Run one trial's election: the primary-narrowing stage when the config
    /// has one, then the main election, then the candidate/candidate utility
    /// covariance. Leaves `ordered_final_cands` holding the final field in
    /// increasing-regret order.
    ///
    /// This is the half of a trial both modes share; what differs is only which
    /// voting methods are then run over the result.
    fn run_election(&mut self) {
        self.itrial += 1;
        log::debug!("{:?} election {}", self.mode, self.itrial);

        if let Some(primary_method) = &mut self.primary_method {
            let sim_primary: &mut Sim = self
                .sim_primary
                .as_mut()
                .expect("primary_method is Some only when sim_primary is");
            sim_primary.election(&mut self.axes, &mut self.rng);
            let final_candidates = primary_method.multi_elect(sim_primary, self.sim.ncand);
            log::debug!("primary election winners: {:?}", final_candidates);
            self.sim.take_from_primary(sim_primary, final_candidates);

            self.ordered_final_cands.clear();
            for &fc in sim_primary.cand_by_regret.iter() {
                if final_candidates.iter().any(|c| c.cand == fc) {
                    self.ordered_final_cands.push(fc);
                }
            }
        } else {
            self.sim.election(&mut self.axes, &mut self.rng);
            self.sim
                .cand_by_regret
                .clone_into(&mut self.ordered_final_cands);
        }

        self.cov_matrix.compute(&self.sim.scores);
        log::debug!("Cov matrix:\n{}", self.cov_matrix.elements);
    }

    /// Run one single-winner trial -- every method in `methods` -- and return
    /// the trial's output row.
    pub(crate) fn do_trial(&mut self) -> ExperimentResult {
        assert_eq!(
            self.mode,
            RunMode::SingleWinner,
            "do_trial is only valid in single-winner mode"
        );
        self.run_election();

        let mut method_results: BTreeMap<String, MethodResult> = BTreeMap::new();
        let mut prev_rslt = None;
        for method in self.methods.iter_mut() {
            let (pairing, result) = method.elect(&self.sim, prev_rslt);
            if let Strategy::Honest = method.method.strat() {
                prev_rslt = Some(pairing);
            }
            // Note here that result.winner is the regret-ranked index of the winner, not the candidate number.
            log::debug!(
                "Method {:?} found winner {} -- regret {}",
                method.method.name(),
                pairing.winner.cand,
                result.regret,
            );
            method_results.insert(method.colname(), result);
        }

        experiment_result(
            &self.sim,
            &self.axes,
            &self.cov_matrix,
            &self.ordered_final_cands,
            method_results,
        )
    }

    /// Run one multi-winner trial -- every method in `committees` -- and return
    /// the trial's output row.
    pub(crate) fn do_committee_trial(&mut self) -> CommitteeResult {
        assert_eq!(
            self.mode,
            RunMode::MultiWinner,
            "do_committee_trial is only valid in multi-winner mode"
        );
        self.run_election();

        let mut method_results: BTreeMap<String, CommitteeMethodResult> = BTreeMap::new();
        for committee in self.committees.iter_mut() {
            let result = committee.elect(&self.sim);
            log::debug!(
                "Method {:?} elected winners (regret ranks) {:?} -- mean regret {}",
                committee.method.name(),
                result.winners,
                result.regret,
            );
            method_results.insert(committee.colname(), result);
        }

        committee_result(&self.sim, &self.axes, &self.cov_matrix, method_results)
    }

    /// Per-method summary statistics accumulated over every trial run so far,
    /// taken from whichever set of trackers this mode populated.
    pub(crate) fn method_stats(&self) -> Vec<SendableMethodReport> {
        match self.mode {
            RunMode::SingleWinner => self.methods.iter().map(|m| m.sendable_report()).collect(),
            RunMode::MultiWinner => self
                .committees
                .iter()
                .map(|c| c.sendable_report())
                .collect(),
        }
    }
}

/// Assemble one trial's [`ExperimentResult`] from the just-run `sim`.
fn experiment_result(
    sim: &Sim,
    axes: &[ConsiderationSimKind],
    cov_matrix: &CovMatrix,
    ordered_final_cands: &[usize],
    methods: BTreeMap<String, MethodResult>,
) -> ExperimentResult {
    use ConsiderationSimKind::*;
    let by_regret = &sim.cand_by_regret;

    let cand_regret = by_regret.iter().map(|&ic| sim.regrets[ic]).collect();
    let in_smith = by_regret.iter().map(|&ic| sim.in_smith_set[ic]).collect();

    // Lower-triangular covariance, reindexed into increasing-regret order.
    let cov = (0..sim.ncand)
        .map(|ix| {
            (0..=ix)
                .map(|iy| cov_matrix.elements[(by_regret[ix], by_regret[iy])])
                .collect()
        })
        .collect();

    let mut likability = None;
    let mut issues = None;
    let mut factions = None;
    for consid in axes {
        match consid {
            Likability(likability_sim) => {
                likability = Some(
                    collect_positions(likability_sim, ordered_final_cands)
                        .into_iter()
                        .map(|coords| coords[0])
                        .collect(),
                );
            }
            Issues(issues_sim) => {
                issues = Some(collect_positions(issues_sim, ordered_final_cands));
            }
            Electorate(factions_sim) => {
                let positions = collect_positions(factions_sim, ordered_final_cands);
                factions = Some(factions_sim.make_faction_info(positions, ordered_final_cands));
            }
            _ => {}
        }
    }

    ExperimentResult {
        ideal_cand: 0,
        cand_regret,
        likability,
        issues,
        electorate: factions,
        cov_matrix: cov,
        num_smith: sim.smith_set_size() as u32,
        in_smith,
        methods,
    }
}

/// Assemble one trial's [`CommitteeResult`] from the just-run `sim`. There's no
/// primary-narrowing stage in multi-winner mode, so (unlike
/// [`experiment_result`]) `sim.cand_by_regret` alone is the right order for
/// both reindexing and indexing into `axes`' candidate positions.
fn committee_result(
    sim: &Sim,
    axes: &[ConsiderationSimKind],
    cov_matrix: &CovMatrix,
    methods: BTreeMap<String, CommitteeMethodResult>,
) -> CommitteeResult {
    use ConsiderationSimKind::*;
    let by_regret = &sim.cand_by_regret;

    let cand_regret = by_regret.iter().map(|&ic| sim.regrets[ic]).collect();
    let in_smith = by_regret.iter().map(|&ic| sim.in_smith_set[ic]).collect();

    // Lower-triangular covariance, reindexed into increasing-regret order.
    let cov = (0..sim.ncand)
        .map(|ix| {
            (0..=ix)
                .map(|iy| cov_matrix.elements[(by_regret[ix], by_regret[iy])])
                .collect()
        })
        .collect();

    let mut likability = None;
    let mut issues = None;
    let mut electorate = None;
    for consid in axes {
        match consid {
            Likability(likability_sim) => {
                likability = Some(
                    collect_positions(likability_sim, by_regret)
                        .into_iter()
                        .map(|coords| coords[0])
                        .collect(),
                );
            }
            Issues(issues_sim) => {
                issues = Some(collect_positions(issues_sim, by_regret));
            }
            Electorate(electorate_sim) => {
                let positions = collect_positions(electorate_sim, by_regret);
                electorate = Some(electorate_sim.make_faction_info(positions, by_regret));
            }
            _ => {}
        }
    }

    CommitteeResult {
        cand_regret,
        likability,
        issues,
        electorate,
        cov_matrix: cov,
        num_smith: sim.smith_set_size() as u32,
        in_smith,
        methods,
    }
}

/// Collect a consideration's candidate positions as `ncand` rows of `dim`
/// coordinates, in `order`. NaN sentinels (used by considerations without a
/// spatial position) are passed through unchanged.
fn collect_positions<CST: ConsiderationSim>(consid: &CST, order: &[usize]) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(order.len());
    let mut current: Vec<f64> = Vec::new();
    consid.push_posn_elements(
        &mut |x, end_of_row| {
            current.push(x);
            if end_of_row {
                rows.push(current.clone());
                current.clear(); // Capacity remains sufficient
            }
        },
        order,
    );
    rows
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// A 3-candidate single-winner config with Likability + Issues(dim 2)
    /// considerations and two methods. `primary` sets `primary_candidates`.
    ///
    /// Shared with [`crate::run`]'s tests so the TOML lives in one place.
    pub(crate) fn single_winner_config(primary: Option<usize>) -> Config {
        let primary_line = primary.map_or(String::new(), |n| format!("primary_candidates = {n}"));
        let toml_str = format!(
            r#"
            voters = 24
            candidates = 3
            {primary_line}

            [[considerations]]
            Likability = {{ mean = 0.5 }}

            [[considerations]]
            [[considerations.Issues]]
            sigma = 1.0
            halfcsep = 0.0
            [[considerations.Issues]]
            sigma = 0.5
            halfcsep = 0.0

            [[methods]]
            Plurality = {{ strat = "Honest" }}

            [[methods]]
            Range = {{ strat = "Honest", nranks = 5 }}
            "#
        );
        toml::from_str(&toml_str).expect("valid single-winner test config")
    }

    /// A 5-candidate multi-winner config (committee size 2), Likability +
    /// Issues considerations, PluralityTopN and RRV as the committee methods.
    ///
    /// Shared with [`crate::run_multi`]'s tests.
    pub(crate) fn multi_winner_config() -> Config {
        let toml_str = r#"
            voters = 24
            candidates = 5
            mode = "multi_winner"
            committee_size = 2

            [[considerations]]
            Likability = { mean = 0.5 }

            [[considerations]]
            [[considerations.Issues]]
            sigma = 1.0
            halfcsep = 0.0

            [[committee_methods]]
            PluralityTopN = {}

            [[committee_methods]]
            [committee_methods.RRV]
            strat = "Honest"
            ranks = 11
            k = 1.0
        "#;
        toml::from_str(toml_str).expect("valid multi-winner test config")
    }

    #[test]
    fn experiment_result_reindexes_sim_state_into_increasing_regret_order() {
        // Post-election state, set by hand. Candidate 1 is best, candidate 0 worst.
        let mut sim = Sim::new(3, 2);
        sim.regrets = vec![2.0, 0.0, 1.0];
        sim.cand_by_regret = vec![1, 2, 0];
        sim.regret_rank = vec![2, 0, 1];
        sim.in_smith_set = vec![true, true, false];

        let mut cov = CovMatrix::new(3);
        for i in 0..3 {
            for j in 0..3 {
                cov.elements[(i, j)] = (10 * i + j) as f64;
            }
        }

        let methods = BTreeMap::from([(
            "pl_h".to_string(),
            MethodResult {
                winner: 2,
                regret: 1.0,
            },
        )]);
        let er = experiment_result(&sim, &[], &cov, &[], methods);

        assert_eq!(er.ideal_cand, 0);
        assert_eq!(er.cand_regret, vec![0.0, 1.0, 2.0]);
        assert_eq!(er.in_smith, vec![true, false, true]);
        assert_eq!(er.num_smith, 2);
        assert!(er.likability.is_none());
        assert!(er.issues.is_none());
        // cov[ix][iy] == elements[(by_regret[ix], by_regret[iy])], by_regret = [1, 2, 0].
        assert_eq!(
            er.cov_matrix,
            vec![vec![11.0], vec![21.0, 22.0], vec![1.0, 2.0, 0.0]]
        );
        assert_eq!(er.methods["pl_h"].winner, 2);
    }

    #[test]
    fn committee_result_reindexes_sim_state_into_increasing_regret_order() {
        let mut sim = Sim::new(3, 2);
        sim.regrets = vec![2.0, 0.0, 1.0];
        sim.cand_by_regret = vec![1, 2, 0];
        sim.regret_rank = vec![2, 0, 1];
        sim.in_smith_set = vec![true, true, false];

        let mut cov = CovMatrix::new(3);
        for i in 0..3 {
            for j in 0..3 {
                cov.elements[(i, j)] = (10 * i + j) as f64;
            }
        }

        let methods = BTreeMap::from([(
            "pltn".to_string(),
            CommitteeMethodResult {
                winners: vec![0, 2],
                regret: 0.5,
            },
        )]);
        let cr = committee_result(&sim, &[], &cov, methods);

        assert_eq!(cr.cand_regret, vec![0.0, 1.0, 2.0]);
        assert_eq!(cr.in_smith, vec![true, false, true]);
        assert_eq!(cr.num_smith, 2);
        assert!(cr.likability.is_none());
        // cov[ix][iy] == elements[(by_regret[ix], by_regret[iy])], by_regret = [1, 2, 0].
        assert_eq!(
            cr.cov_matrix,
            vec![vec![11.0], vec![21.0, 22.0], vec![1.0, 2.0, 0.0]]
        );
        assert_eq!(cr.methods["pltn"].winners, vec![0, 2]);
    }

    #[test]
    fn collect_positions_yields_one_row_of_coords_per_candidate() {
        let mut sim = Sim::new(3, 40);
        let config = single_winner_config(None);
        let mut axes: Vec<ConsiderationSimKind> = config
            .considerations
            .iter()
            .map(|c| c.new_sim(&sim))
            .collect();
        // Choice positions are RNG-generated during the election.
        sim.election(&mut axes, &mut rand::rng());

        // axes[0] is Likability (1 value per candidate).
        let likability = collect_positions(&axes[0], &sim.cand_by_regret);
        assert_eq!(likability.len(), 3);
        assert!(likability.iter().all(|coords| coords.len() == 1));

        // axes[1] is Issues with 2 axes -> 2 coordinates per candidate.
        let issues = collect_positions(&axes[1], &sim.cand_by_regret);
        assert_eq!(issues.len(), 3);
        assert!(issues.iter().all(|coords| coords.len() == 2));
        assert_ne!(issues[0], issues[1]);
    }

    /// Repeated `do_trial` calls on one reused runner are driven only by the
    /// RNG, so a seeded run replays exactly.
    #[test]
    fn runner_do_trial_replays_with_a_seeded_rng() {
        let config = single_winner_config(None);
        let trials = |seed: u64| {
            let mut runner = TrialRunner::new(&config, StdRng::seed_from_u64(seed));
            (0..4).map(|_| runner.do_trial()).collect::<Vec<_>>()
        };

        let first = trials(0x5EED);
        let again = trials(0x5EED);
        assert_eq!(first.len(), 4);
        for (a, b) in first.iter().zip(&again) {
            assert_eq!(a.cand_regret, b.cand_regret);
            assert_eq!(a.methods["pl_h"].winner, b.methods["pl_h"].winner);
            assert_eq!(a.methods["range_5_h"].regret, b.methods["range_5_h"].regret);
        }
        // Successive trials advance the RNG rather than repeating themselves.
        assert_ne!(first[0].cand_regret, first[1].cand_regret);
    }

    /// The multi-winner half of the same guarantee.
    #[test]
    fn runner_do_committee_trial_replays_with_a_seeded_rng() {
        let config = multi_winner_config();
        let trials = |seed: u64| {
            let mut runner = TrialRunner::new(&config, StdRng::seed_from_u64(seed));
            (0..4)
                .map(|_| runner.do_committee_trial())
                .collect::<Vec<_>>()
        };

        let first = trials(0xC0FFEE);
        let again = trials(0xC0FFEE);
        assert_eq!(first.len(), 4);
        for (a, b) in first.iter().zip(&again) {
            assert_eq!(a.cand_regret, b.cand_regret);
            assert_eq!(a.methods["pltn"].winners, b.methods["pltn"].winners);
            assert_eq!(a.methods["rrv_11_h"].regret, b.methods["rrv_11_h"].regret);
        }
        assert_ne!(first[0].cand_regret, first[1].cand_regret);
    }

    #[test]
    fn a_primary_narrows_the_field_before_the_single_winner_round() {
        let config = single_winner_config(Some(6));
        let mut runner = TrialRunner::new(&config, StdRng::seed_from_u64(1));
        let er = runner.do_trial();

        // 3 finalists reported, drawn from the 6-candidate primary field.
        assert_eq!(er.cand_regret.len(), 3);
        assert_eq!(er.methods.len(), 2);
        assert_eq!(runner.ordered_final_cands.len(), 3);
        assert!(runner.ordered_final_cands.iter().all(|&c| c < 6));
    }

    /// Multi-winner mode builds no `methods` trackers, so asking for a
    /// single-winner trial is a programming error rather than an empty row.
    #[test]
    #[should_panic(expected = "only valid in single-winner mode")]
    fn do_trial_rejects_a_multi_winner_runner() {
        let config = multi_winner_config();
        TrialRunner::new(&config, StdRng::seed_from_u64(1)).do_trial();
    }

    /// A multi-winner config never builds the primary stage, even when
    /// `primary_candidates` is left over in the config.
    #[test]
    fn multi_winner_mode_ignores_primary_candidates() {
        let mut config = multi_winner_config();
        config.primary_candidates = Some(9);
        let runner = TrialRunner::new(&config, StdRng::seed_from_u64(1));

        assert!(runner.sim_primary.is_none());
        assert!(runner.primary_method.is_none());
        assert_eq!(runner.sim.ncand, 5);
    }
}
