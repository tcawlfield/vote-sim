// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! One worker's state for a batch of trials, shared by both election modes.
//!
//! [`TrialRunner::new`] allocates everything a batch needs once. Each call to
//! [`TrialRunner::do_trial`] (single-winner) or
//! [`TrialRunner::do_committee_trial`] (multi-winner) then runs one election
//! and returns that trial's output row, reusing all of it.

use std::collections::BTreeMap;

use rand::Rng;

use crate::committee_welfare::WelfareEval;
use crate::config::{Config, RunMode};
use crate::considerations::{ConsiderationSim, ConsiderationSimKind};
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::{CommitteeTracker, MethodTracker, SendableMethodReport};
use crate::methods::{MWMethodSim, Method, WinnerAndRunnerup};
use crate::out_types::{
    CommitteeMethodResult, CommitteeResult, ExperimentResult, MethodResult, WelfareBounds,
};
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
    /// Single-winner methods, as laid out by [`method_trackers`]. Non-empty
    /// only in [`RunMode::SingleWinner`].
    methods: Vec<MethodTracker>,
    /// Multi-winner methods. Non-empty only in [`RunMode::MultiWinner`].
    committees: Vec<CommitteeTracker>,
    /// Judges every committee by the config's `committee_welfare`. `Some` only
    /// in [`RunMode::MultiWinner`], and only when that list isn't empty.
    welfare: Option<WelfareEval>,
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
    /// Saved results for strategic methods' pre-election polls.
    method_poll_pairings: Vec<WinnerAndRunnerup>,
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

        let mut welfare = None;
        let (methods, committees): (Vec<MethodTracker>, Vec<CommitteeTracker>) = match config.mode {
            RunMode::SingleWinner => (method_trackers(&config.methods, &sim), Vec::new()),
            RunMode::MultiWinner => {
                let committee_size = config
                    .committee_size
                    .expect("Config::validate ensures committee_size is set in MultiWinner mode");
                if !config.committee_welfare.is_empty() {
                    welfare = Some(WelfareEval::new(
                        &config.committee_welfare,
                        &sim,
                        committee_size,
                    ));
                }
                (
                    Vec::new(),
                    config
                        .committee_methods
                        .iter()
                        .map(|m| {
                            CommitteeTracker::new(
                                m,
                                &sim,
                                committee_size,
                                &config.committee_welfare,
                            )
                        })
                        .collect(),
                )
            }
        };

        let cov_matrix = CovMatrix::new(sim.ncand);

        let primary_method = sim_primary
            .as_ref()
            .map(|sim_primary| config.primary_method.new_sim(sim_primary));

        let ordered_final_cands = vec![0; sim.ncand];
        let method_poll_pairings = Vec::with_capacity(methods.len());

        TrialRunner {
            mode: config.mode,
            rng,
            sim,
            sim_primary,
            axes,
            methods,
            committees,
            welfare,
            cov_matrix,
            primary_method,
            ordered_final_cands,
            itrial: 0,
            method_poll_pairings,
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
        // This trial's winner/runner-up for each tracker so far, which later
        // strategic methods look up by index as their pre-election poll.
        self.method_poll_pairings.clear();
        for method in self.methods.iter_mut() {
            let poll = method.poll.map(|i| self.method_poll_pairings[i]);
            let (pairing, result) = method.elect(&self.sim, poll);
            self.method_poll_pairings.push(pairing);
            if !method.reported {
                continue;
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
        if let Some(welfare) = &mut self.welfare {
            welfare.prepare(&self.sim);
        }

        let mut method_results: BTreeMap<String, CommitteeMethodResult> = BTreeMap::new();
        for committee in self.committees.iter_mut() {
            let result = committee.elect(&self.sim, self.welfare.as_mut());
            log::debug!(
                "Method {:?} elected winners (regret ranks) {:?} -- mean regret {}",
                committee.method.name(),
                result.winners,
                result.regret,
            );
            method_results.insert(committee.colname(), result);
        }

        let welfare_bounds = self.welfare.as_ref().map(|eval| {
            eval.colnames()
                .iter()
                .zip(eval.best.iter().zip(&eval.mean))
                .map(|(colname, (&best, &mean))| (colname.clone(), WelfareBounds { best, mean }))
                .collect()
        });
        committee_result(
            &self.sim,
            &self.axes,
            &self.cov_matrix,
            method_results,
            welfare_bounds,
        )
    }

    /// Per-method summary statistics accumulated over every trial run so far,
    /// taken from whichever set of trackers this mode populated.
    pub(crate) fn method_stats(&self) -> Vec<SendableMethodReport> {
        match self.mode {
            RunMode::SingleWinner => self
                .methods
                .iter()
                .filter(|m| m.reported)
                .map(|m| m.sendable_report())
                .collect(),
            RunMode::MultiWinner => self
                .committees
                .iter()
                .map(|c| c.sendable_report())
                .collect(),
        }
    }
}

/// Build the single-winner trackers for the configured `methods`, keeping config
/// order but making sure every strategic method's honest poll (see
/// [`Method::honest_poll`]) runs somewhere before it.
///
/// Methods are matched by `Method` equality. A poll the config also lists is
/// pulled forward if need be and reported as usual; one it doesn't list is
/// added unreported, purely as input to the strategic method. A method listed
/// twice fails [`Config::validate`]; if one gets here anyway, it runs once.
fn method_trackers(methods: &[Method], sim: &Sim) -> Vec<MethodTracker> {
    let mut trackers: Vec<MethodTracker> = Vec::with_capacity(methods.len());
    // The method behind each tracker, in the same order.
    let mut built: Vec<Method> = Vec::with_capacity(methods.len());
    for method in methods {
        if built.contains(method) {
            continue; // pulled forward as an earlier method's poll, or listed twice
        }
        let poll = method.honest_poll().map(|poll| {
            built.iter().position(|m| *m == poll).unwrap_or_else(|| {
                // Build a configured poll from its own entry, which carries its
                // `colname`; the twin compares equal but has the default name.
                let (poll, reported) = match methods.iter().find(|m| **m == poll) {
                    Some(configured) => (configured.clone(), true),
                    None => (poll, false),
                };
                trackers.push(MethodTracker::new(&poll, sim, reported, None));
                built.push(poll);
                trackers.len() - 1
            })
        });
        trackers.push(MethodTracker::new(method, sim, true, poll));
        built.push(method.clone());
    }
    trackers
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
    welfare: Option<BTreeMap<String, WelfareBounds>>,
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
        welfare,
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
                welfare: None,
            },
        )]);
        let cr = committee_result(&sim, &[], &cov, methods, None);

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
    fn committee_welfare_is_left_out_unless_configured() {
        let config = multi_winner_config();
        assert!(config.committee_welfare.is_empty());
        let mut runner = TrialRunner::new(&config, StdRng::seed_from_u64(1));
        assert!(runner.welfare.is_none());
        let result = runner.do_committee_trial();
        assert!(result.welfare.is_none());
        assert!(result.methods.values().all(|m| m.welfare.is_none()));
    }

    #[test]
    fn committee_welfare_judges_every_method_against_every_committee() {
        use crate::committee_welfare::Welfare;
        let mut config = multi_winner_config();
        config.committee_welfare = vec![Welfare::Additive, Welfare::Harmonic];
        let mut runner = TrialRunner::new(&config, StdRng::seed_from_u64(2));
        for _ in 0..5 {
            let result = runner.do_committee_trial();
            let bounds = result.welfare.expect("welfare bounds when configured");
            assert_eq!(bounds.keys().collect::<Vec<_>>(), ["add", "pav"]);
            for b in bounds.values() {
                assert!(b.best >= b.mean, "{b:?}");
            }
            for (colname, method) in &result.methods {
                let welfare = method.welfare.as_ref().expect("per-method welfare");
                assert_eq!(welfare.keys().collect::<Vec<_>>(), ["add", "pav"]);
                for (name, w) in welfare {
                    assert!(w.value <= bounds[name].best + 1e-12, "{colname} {name}");
                    assert!(w.regret >= -1e-12, "{colname} {name}");
                }
            }
        }
        let stats = runner.method_stats();
        assert_eq!(stats.len(), 2);
        for report in stats {
            let names: Vec<&str> = report
                .welfare_regret
                .iter()
                .map(|(n, _)| n.as_str())
                .collect();
            assert_eq!(names, ["add", "pav"]);
        }
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

    fn method(json: &str) -> Method {
        serde_json::from_str(json).expect("valid Method JSON")
    }

    /// `(colname, reported, poll)` for each tracker, in run order.
    fn layout(methods: &[Method]) -> Vec<(String, bool, Option<usize>)> {
        method_trackers(methods, &Sim::new(3, 5))
            .iter()
            .map(|t| (t.colname(), t.reported, t.poll))
            .collect()
    }

    fn row(colname: &str, reported: bool, poll: Option<usize>) -> (String, bool, Option<usize>) {
        (colname.to_string(), reported, poll)
    }

    const PL_H: &str = r#"{"Plurality": {"strat": "Honest"}}"#;
    const PL_S: &str = r#"{"Plurality": {"strat": "Strategic"}}"#;
    const RANGE_H: &str = r#"{"Range": {"strat": "Honest", "nranks": 10}}"#;
    const RANGE_S: &str = r#"{"Range": {"strat": "Strategic", "nranks": 10}}"#;

    #[test]
    fn a_lone_strategic_method_gets_a_hidden_honest_poll() {
        assert_eq!(
            layout(&[method(PL_S)]),
            [row("pl_h", false, None), row("pl_s", true, Some(0))]
        );
    }

    #[test]
    fn an_honest_poll_listed_later_is_pulled_forward_and_still_reported() {
        assert_eq!(
            layout(&[method(PL_S), method(PL_H)]),
            [row("pl_h", true, None), row("pl_s", true, Some(0))]
        );
    }

    #[test]
    fn an_honest_poll_listed_earlier_is_reused_in_place() {
        assert_eq!(
            layout(&[method(RANGE_H), method(PL_H), method(RANGE_S)]),
            [
                row("range_10_h", true, None),
                row("pl_h", true, None),
                row("range_10_s", true, Some(0)),
            ]
        );
    }

    /// The old rule polled from whichever honest method ran last, so this
    /// strategic Range used to take honest Plurality's result as its poll.
    #[test]
    fn a_strategic_method_polls_its_own_honest_twin_not_the_nearest_honest_one() {
        assert_eq!(
            layout(&[method(PL_H), method(RANGE_S)]),
            [
                row("pl_h", true, None),
                row("range_10_h", false, None),
                row("range_10_s", true, Some(1)),
            ]
        );
    }

    #[test]
    fn strategic_methods_that_take_no_poll_get_none() {
        let multivote = r#"{"Multivote": {"strat": "Strategic", "votes": 3, "spread_fact": 1.0}}"#;
        assert_eq!(
            layout(&[method(multivote)]),
            [row("multi_s_3v", true, None)]
        );
    }

    /// The honest poll drops strategic STAR's non-default stretch factor, so
    /// it matches the plainly-configured honest STAR instead of duplicating it.
    #[test]
    fn strategy_only_parameters_dont_prevent_a_poll_match() {
        let star_h = r#"{"STAR": {"strat": "Honest"}}"#;
        let star_s = r#"{"STAR": {"strat": "Strategic", "strategic_stretch_factor": 2.0}}"#;
        assert_eq!(
            layout(&[method(star_h), method(star_s)]),
            [row("star_6_h", true, None), row("star_6_s", true, Some(0))]
        );
    }

    /// Polls are matched by `Method` equality, which ignores `colname`: a
    /// renamed honest method still serves as the poll, under its own name.
    #[test]
    fn a_renamed_honest_method_still_serves_as_the_poll() {
        let pl_h_renamed = r#"{"Plurality": {"strat": "Honest", "colname": "first_choice"}}"#;
        assert_eq!(
            layout(&[method(PL_S), method(pl_h_renamed)]),
            [row("first_choice", true, None), row("pl_s", true, Some(0))]
        );
    }

    /// A hidden poll is built from the strategic method's parameters, but not
    /// its name: that belongs to the strategic method's own column.
    #[test]
    fn a_hidden_poll_takes_the_default_name_not_the_strategic_ones() {
        let pl_s_renamed = r#"{"Plurality": {"strat": "Strategic", "colname": "pl_tactical"}}"#;
        assert_eq!(
            layout(&[method(pl_s_renamed)]),
            [row("pl_h", false, None), row("pl_tactical", true, Some(0))]
        );
    }

    /// `Config::validate` rejects this config; `method_trackers` just doesn't
    /// build a second copy.
    #[test]
    fn a_method_listed_twice_runs_once() {
        assert_eq!(
            layout(&[method(PL_H), method(PL_H)]),
            [row("pl_h", true, None)]
        );
    }

    /// Strategic Range used to `unwrap()` a missing poll and panic. Now it runs,
    /// and only the configured methods reach the output and the summary.
    #[test]
    fn do_trial_reports_only_configured_methods_when_polls_are_hidden() {
        let mut config = single_winner_config(None);
        config.methods = vec![method(PL_S), method(RANGE_S)];
        let mut runner = TrialRunner::new(&config, StdRng::seed_from_u64(3));
        assert_eq!(runner.methods.len(), 4);

        for _ in 0..5 {
            let er = runner.do_trial();
            assert_eq!(
                er.methods.keys().collect::<Vec<_>>(),
                ["pl_s", "range_10_s"]
            );
        }
        let stats: Vec<String> = runner
            .method_stats()
            .iter()
            .map(|r| r.name.clone())
            .collect();
        assert_eq!(stats, ["Plurality, Strategic", "Range 1-10, Strategic"]);
        let colnames: Vec<String> = runner
            .method_stats()
            .into_iter()
            .map(|r| r.colname)
            .collect();
        assert_eq!(colnames, ["pl_s", "range_10_s"]);
    }
}
