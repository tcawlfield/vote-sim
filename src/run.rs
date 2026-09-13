// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::sync::mpsc;

use arrow_array::RecordBatch;
use parquet::file::metadata::KeyValue;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use work_queue::Queue;

use crate::config::{Config, RunMode};
use crate::considerations::{ConsiderationSim, ConsiderationSimKind};
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::{CommitteeTracker, MethodTracker, SendableMethodReport};
use crate::methods::Strategy;
use crate::out_types::{CommitteeMethodResult, CommitteeResult, ExperimentResult, MethodResult};
use crate::sim::Sim;

static MAX_TRIALS_PER_JOB: usize = 10000;

struct Task {
    config: Config,
    trials: usize,
    result_chan: mpsc::Sender<TaskResult>,
}

struct TaskResult {
    method_stats: Vec<SendableMethodReport>,
    results: Vec<ExperimentResult>,
}

struct CommitteeTask {
    config: Config,
    trials: usize,
    result_chan: mpsc::Sender<CommitteeTaskResult>,
}

struct CommitteeTaskResult {
    method_stats: Vec<SendableMethodReport>,
    results: Vec<CommitteeResult>,
}

pub fn run_sims(
    config: &Config,
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
) -> Result<(), Box<dyn Error>> {
    config.validate()?;
    match config.mode {
        RunMode::SingleWinner => run_sims_single_winner(config, trials, outfile),
        RunMode::MultiWinner => run_sims_multi_winner(config, trials, outfile),
    }
}

fn run_sims_single_winner(
    config: &Config,
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
) -> Result<(), Box<dyn Error>> {
    let num_workers = std::thread::available_parallelism().unwrap().get();
    let min_chunks = num_workers.max(trials.div_ceil(MAX_TRIALS_PER_JOB));
    let chunks_per_worker = min_chunks.div_ceil(num_workers);
    let chunks = chunks_per_worker * num_workers;
    let trials_per_chunk = (trials + 1) / chunks;
    log::info!(
        "{} worker threads, {} batches of about {} events",
        num_workers,
        chunks,
        trials_per_chunk
    );

    let (task_result_tx, task_result_rx) = mpsc::channel();

    let queue: Queue<Task> = Queue::new(num_workers, 4);
    let mut trials_left = trials;
    for chunks_to_do in (1..chunks + 1).rev() {
        let task_trials = trials_left.div_ceil(chunks_to_do);
        let task = Task {
            config: config.clone(),
            trials: task_trials,
            result_chan: task_result_tx.clone(),
        };
        queue.push(task);
        trials_left -= task_trials;
    }

    let _handles: Vec<_> = queue
        .local_queues()
        .map(|mut local_queue| {
            std::thread::spawn(move || {
                while let Some(task) = local_queue.pop() {
                    run_batch(&task.config, task.trials, &task.result_chan).unwrap();
                }
            })
        })
        .collect();
    drop(task_result_tx);

    // Collect every worker's trial results into one growing Vec, and fold the
    // per-method summary statistics together as batches arrive.
    let mut all_results: Vec<ExperimentResult> = Vec::with_capacity(trials);
    let mut summaries: Option<Vec<SendableMethodReport>> = None;
    while let Ok(task_result) = task_result_rx.recv() {
        if task_result.results.is_empty() {
            continue;
        }
        log::info!(
            "Completed a batch of {} elections",
            task_result.results.len()
        );
        all_results.extend(task_result.results);
        match summaries.as_mut() {
            Some(summaries) => {
                for (whole, part) in summaries.iter_mut().zip(task_result.method_stats.iter()) {
                    whole.combine(part);
                }
            }
            None => summaries = Some(task_result.method_stats),
        }
    }

    if let Some(filename) = outfile
        && !all_results.is_empty()
    {
        let batch = ExperimentResult::to_record_batch(&all_results);
        let mut writer = get_writer(config, filename, &batch);
        writer.write(&batch)?;
        writer.close()?; // writer must be closed to write the footer
        println!("Wrote {}", filename.to_str().unwrap());
    }

    if let Some(summaries) = summaries {
        for method_report in summaries {
            method_report.report();
        }
    }

    Ok(())
}

fn run_batch(
    config: &Config,
    trials: usize,
    task_result_tx: &mpsc::Sender<TaskResult>,
) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::rng();

    let ncand = config.candidates;
    let nvtr = config.voters;

    let mut sim = Sim::new(ncand, nvtr);
    let mut sim_primary = config.primary_candidates.map(|pcand| Sim::new(pcand, nvtr));

    let mut axes: Vec<ConsiderationSimKind> = {
        let max_sim = sim_primary.as_ref().unwrap_or(&sim);
        config
            .considerations
            .iter()
            .map(|c| c.new_sim(max_sim))
            .collect()
    };

    let mut methods: Vec<MethodTracker> = config
        .methods
        .iter()
        .map(|m| MethodTracker::new(m, &sim))
        .collect();

    let mut cov_matrix = CovMatrix::new(sim.ncand);

    let mut mwms = sim_primary
        .as_ref()
        .map(|sim_primary| config.primary_method.new_sim(sim_primary));

    // ordered_final_cands is a list of candidates in order of increasing regret.
    // With no primary, it is identical to sim.cand_by_regret. With a primary, it
    // contains only the winning primary candidates.
    let mut ordered_final_cands = vec![0; sim.ncand];

    let mut results: Vec<ExperimentResult> = Vec::with_capacity(trials);

    for itrial in 0..trials {
        log::debug!("Sim election {}", itrial + 1);

        if let Some(rrv) = &mut mwms {
            let sim_primary: &mut Sim = sim_primary.as_mut().unwrap();
            sim_primary.election(&mut axes, &mut rng);
            let final_candidates = rrv.multi_elect(sim_primary, sim.ncand);
            log::debug!("primary election winners: {:?}", final_candidates);
            sim.take_from_primary(sim_primary, final_candidates);

            ordered_final_cands.clear();
            for &fc in sim_primary.cand_by_regret.iter() {
                if final_candidates.iter().any(|c| c.cand == fc) {
                    ordered_final_cands.push(fc);
                }
            }
        } else {
            sim.election(&mut axes, &mut rng);
            sim.cand_by_regret.clone_into(&mut ordered_final_cands);
        };

        cov_matrix.compute(&sim.scores);
        log::debug!("Cov matrix:\n{}", cov_matrix.elements);

        let mut method_results: BTreeMap<String, MethodResult> = BTreeMap::new();
        let mut prev_rslt = None;
        for method in methods.iter_mut() {
            let (pairing, result) = method.elect(&sim, prev_rslt);
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

        results.push(experiment_result(
            &sim,
            &axes,
            &cov_matrix,
            &ordered_final_cands,
            method_results,
        ));
    }

    let method_stats = methods.iter().map(|m| m.sendable_report()).collect();
    task_result_tx
        .send(TaskResult {
            method_stats,
            results,
        })
        .expect("Could not send batch results");

    Ok(())
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
                // rows.push(std::mem::take(&mut current));
                rows.push(current.clone());
                current.clear(); // Capacity remains sufficient
            }
        },
        order,
    );
    rows
}

fn run_sims_multi_winner(
    config: &Config,
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
) -> Result<(), Box<dyn Error>> {
    let num_workers = std::thread::available_parallelism().unwrap().get();
    let min_chunks = num_workers.max(trials.div_ceil(MAX_TRIALS_PER_JOB));
    let chunks_per_worker = min_chunks.div_ceil(num_workers);
    let chunks = chunks_per_worker * num_workers;
    let trials_per_chunk = (trials + 1) / chunks;
    log::info!(
        "{} worker threads, {} batches of about {} committee elections",
        num_workers,
        chunks,
        trials_per_chunk
    );

    let (task_result_tx, task_result_rx) = mpsc::channel();

    let queue: Queue<CommitteeTask> = Queue::new(num_workers, 4);
    let mut trials_left = trials;
    for chunks_to_do in (1..chunks + 1).rev() {
        let task_trials = trials_left.div_ceil(chunks_to_do);
        let task = CommitteeTask {
            config: config.clone(),
            trials: task_trials,
            result_chan: task_result_tx.clone(),
        };
        queue.push(task);
        trials_left -= task_trials;
    }

    let _handles: Vec<_> = queue
        .local_queues()
        .map(|mut local_queue| {
            std::thread::spawn(move || {
                while let Some(task) = local_queue.pop() {
                    run_batch_multi_winner(&task.config, task.trials, &task.result_chan).unwrap();
                }
            })
        })
        .collect();
    drop(task_result_tx);

    let mut all_results: Vec<CommitteeResult> = Vec::with_capacity(trials);
    let mut summaries: Option<Vec<SendableMethodReport>> = None;
    while let Ok(task_result) = task_result_rx.recv() {
        if task_result.results.is_empty() {
            continue;
        }
        log::info!(
            "Completed a batch of {} committee elections",
            task_result.results.len()
        );
        all_results.extend(task_result.results);
        match summaries.as_mut() {
            Some(summaries) => {
                for (whole, part) in summaries.iter_mut().zip(task_result.method_stats.iter()) {
                    whole.combine(part);
                }
            }
            None => summaries = Some(task_result.method_stats),
        }
    }

    if let Some(filename) = outfile
        && !all_results.is_empty()
    {
        let batch = CommitteeResult::to_record_batch(&all_results);
        let mut writer = get_writer(config, filename, &batch);
        writer.write(&batch)?;
        writer.close()?; // writer must be closed to write the footer
        println!("Wrote {}", filename.to_str().unwrap());
    }

    if let Some(summaries) = summaries {
        for method_report in summaries {
            method_report.report();
        }
    }

    Ok(())
}

fn run_batch_multi_winner(
    config: &Config,
    trials: usize,
    task_result_tx: &mpsc::Sender<CommitteeTaskResult>,
) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::rng();

    let ncand = config.candidates;
    let nvtr = config.voters;
    let committee_size = config
        .committee_size
        .expect("Config::validate ensures committee_size is set in MultiWinner mode");

    let mut sim = Sim::new(ncand, nvtr);
    let mut axes: Vec<ConsiderationSimKind> = config
        .considerations
        .iter()
        .map(|c| c.new_sim(&sim))
        .collect();

    let mut committees: Vec<CommitteeTracker> = config
        .committee_methods
        .iter()
        .map(|m| CommitteeTracker::new(m, &sim, committee_size))
        .collect();

    let mut cov_matrix = CovMatrix::new(sim.ncand);
    let mut results: Vec<CommitteeResult> = Vec::with_capacity(trials);

    for itrial in 0..trials {
        log::debug!("Committee election {}", itrial + 1);
        sim.election(&mut axes, &mut rng);
        cov_matrix.compute(&sim.scores);

        let mut method_results: BTreeMap<String, CommitteeMethodResult> = BTreeMap::new();
        for committee in committees.iter_mut() {
            let result = committee.elect(&sim);
            log::debug!(
                "Method {:?} elected winners (regret ranks) {:?} -- mean regret {}",
                committee.method.name(),
                result.winners,
                result.regret,
            );
            method_results.insert(committee.colname(), result);
        }

        results.push(committee_result(&sim, &axes, &cov_matrix, method_results));
    }

    let method_stats = committees.iter().map(|c| c.sendable_report()).collect();
    task_result_tx
        .send(CommitteeTaskResult {
            method_stats,
            results,
        })
        .expect("Could not send batch results");

    Ok(())
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

fn get_writer(
    config: &Config,
    filename: &std::ffi::OsStr,
    sample_batch: &RecordBatch,
) -> ArrowWriter<fs::File> {
    let config_str = serde_json::to_string(config).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "voting_config".to_owned(),
            config_str,
        )]))
        .build();
    let file = fs::File::create(filename).unwrap();
    ArrowWriter::try_new(file, sample_batch.schema(), Some(props)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    /// A 3-candidate config with a Likability + Issues(dim 2) consideration and
    /// two methods. `primary` sets `primary_candidates` when `Some`.
    fn test_config(primary: Option<usize>) -> Config {
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
        toml::from_str(&toml_str).expect("valid test config")
    }

    fn run_one_batch(config: &Config, trials: usize) -> TaskResult {
        let (tx, rx) = mpsc::channel();
        run_batch(config, trials, &tx).unwrap();
        drop(tx);
        rx.recv().unwrap()
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
    fn collect_positions_yields_one_row_of_coords_per_candidate() {
        let mut sim = Sim::new(3, 40);
        let config = test_config(None);
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

    #[test]
    fn run_batch_emits_one_well_formed_result_per_trial() {
        let config = test_config(None);
        let result = run_one_batch(&config, 9);

        assert_eq!(result.results.len(), 9);
        assert_eq!(result.method_stats.len(), 2);
        for er in &result.results {
            assert_eq!(er.cand_regret.len(), 3);
            assert_eq!(er.in_smith.len(), 3);
            assert_eq!(er.cov_matrix.len(), 3);
            assert_eq!(er.methods.len(), 2);
            assert_eq!(er.cand_regret[0], 0.0); // best candidate has zero regret
            assert_eq!(er.likability.as_ref().unwrap().len(), 3);
            assert_eq!(er.issues.as_ref().unwrap().len(), 3);
            assert!(er.issues.as_ref().unwrap().iter().all(|c| c.len() == 2));
        }
    }

    #[test]
    fn run_batch_with_a_primary_still_reports_the_final_field() {
        let config = test_config(Some(6));
        let result = run_one_batch(&config, 5);

        assert_eq!(result.results.len(), 5);
        for er in &result.results {
            assert_eq!(er.cand_regret.len(), 3); // 3 finalists, not 6 primary candidates
            assert_eq!(er.methods.len(), 2);
        }
    }

    #[test]
    fn run_sims_without_an_outfile_completes() {
        run_sims(&test_config(None), 20, &None).unwrap();
    }

    #[test]
    fn run_sims_writes_a_parquet_that_reads_back() {
        let config = test_config(None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trials.parquet");

        run_sims(&config, 30, &Some(path.clone().into_os_string())).unwrap();

        let file = fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

        let column_names: Vec<String> = builder
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            column_names,
            [
                "ideal_cand",
                "cand_regret",
                "likability",
                "issues",
                "electorate",
                "cov_matrix",
                "num_smith",
                "in_smith",
                "methods"
            ]
        );

        let has_config_meta = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .is_some_and(|kvs| kvs.iter().any(|kv| kv.key == "voting_config"));
        assert!(has_config_meta);

        let rows: usize = builder
            .build()
            .unwrap()
            .map(|b| b.unwrap().num_rows())
            .sum();
        assert_eq!(rows, 30);
    }

    /// A 5-candidate MultiWinner config (committee size 2), Likability + Issues
    /// considerations, PluralityTopN and RRV as the committee methods.
    fn test_committee_config() -> Config {
        let toml_str = r#"
            voters = 24
            candidates = 5
            mode = "MultiWinner"
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
        toml::from_str(toml_str).expect("valid committee test config")
    }

    fn run_one_committee_batch(config: &Config, trials: usize) -> CommitteeTaskResult {
        let (tx, rx) = mpsc::channel();
        run_batch_multi_winner(config, trials, &tx).unwrap();
        drop(tx);
        rx.recv().unwrap()
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
    fn run_batch_multi_winner_emits_one_well_formed_committee_per_trial() {
        let config = test_committee_config();
        let result = run_one_committee_batch(&config, 9);

        assert_eq!(result.results.len(), 9);
        assert_eq!(result.method_stats.len(), 2);
        for cr in &result.results {
            assert_eq!(cr.cand_regret.len(), 5);
            assert_eq!(cr.in_smith.len(), 5);
            assert_eq!(cr.cov_matrix.len(), 5);
            assert_eq!(cr.cand_regret[0], 0.0); // best candidate has zero regret
            assert_eq!(cr.likability.as_ref().unwrap().len(), 5);
            assert_eq!(cr.issues.as_ref().unwrap().len(), 5);

            assert_eq!(cr.methods.len(), 2);
            for method in cr.methods.values() {
                assert_eq!(method.winners.len(), 2); // committee_size
            }
        }
    }

    #[test]
    fn run_sims_dispatches_to_multi_winner_mode_and_writes_a_readable_parquet() {
        let config = test_committee_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("committees.parquet");

        run_sims(&config, 15, &Some(path.clone().into_os_string())).unwrap();

        let file = fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let column_names: Vec<String> = builder
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            column_names,
            [
                "cand_regret",
                "likability",
                "issues",
                "electorate",
                "cov_matrix",
                "num_smith",
                "in_smith",
                "methods"
            ]
        );

        let rows: usize = builder
            .build()
            .unwrap()
            .map(|b| b.unwrap().num_rows())
            .sum();
        assert_eq!(rows, 15);
    }

    #[test]
    fn run_sims_rejects_an_invalid_config_instead_of_panicking() {
        let mut config = test_committee_config();
        config.committee_size = None;
        assert!(run_sims(&config, 5, &None).is_err());
    }
}
