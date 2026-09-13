// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::sync::mpsc;

use arrow_array::RecordBatch;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use work_queue::Queue;

use crate::config::{Config, RunMode};
use crate::considerations::{ConsiderationSim, ConsiderationSimKind};
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::{CommitteeTracker, MethodTracker, SendableMethodReport};
use crate::methods::Strategy;
use crate::out_types::{CommitteeMethodResult, CommitteeResult, ExperimentResult, MethodResult};
use crate::run::{collect_positions, get_writer};
use crate::sim::Sim;

static MAX_TRIALS_PER_JOB: usize = 10000;

struct CommitteeTask {
    config: Config,
    trials: usize,
    result_chan: mpsc::Sender<CommitteeTaskResult>,
}

struct CommitteeTaskResult {
    method_stats: Vec<SendableMethodReport>,
    results: Vec<CommitteeResult>,
}

pub fn run_sims_multi_winner(
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

#[cfg(test)]
mod tests {
    use crate::run_sims;

    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    /// A 5-candidate MultiWinner config (committee size 2), Likability + Issues
    /// considerations, PluralityTopN and RRV as the committee methods.
    fn test_committee_config() -> Config {
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
