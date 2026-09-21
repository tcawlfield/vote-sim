// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fs;
use std::sync::mpsc;

use arrow_array::RecordBatch;
use parquet::file::metadata::KeyValue;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use work_queue::Queue;

use crate::config::{Config, RunMode};
use crate::method_tracker::SendableMethodReport;
use crate::out_types::{CommitteeResult, ExperimentResult};
use crate::run_multi::{collect_multi_winner, run_sims_multi_winner};
use crate::runner::TrialRunner;

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

/// Run `trials` elections and hand back the results as a single Arrow
/// [`RecordBatch`], without touching the filesystem or printing anything.
///
/// The column set depends on `config.mode`: [`ExperimentResult`]'s for
/// `SingleWinner`, [`CommitteeResult`](crate::out_types::CommitteeResult)'s for
/// `MultiWinner`. Returns `None` when `trials` is zero, since there are no rows
/// to infer a schema from.
///
/// This is the entry point for embedders (the Python bindings, notably);
/// [`run_sims`] is the file-writing, stats-reporting command-line path. Worker
/// threads are spawned internally, so callers holding a lock -- such as the
/// Python GIL -- should release it around this call.
pub fn simulate(
    config: &Config,
    trials: usize,
) -> Result<Option<RecordBatch>, Box<dyn Error + Send + Sync>> {
    config.validate()?;
    Ok(match config.mode {
        RunMode::SingleWinner => {
            let (results, _) = collect_single_winner(config, trials);
            (!results.is_empty()).then(|| ExperimentResult::to_record_batch(&results))
        }
        RunMode::MultiWinner => {
            let (results, _) = collect_multi_winner(config, trials);
            (!results.is_empty()).then(|| CommitteeResult::to_record_batch(&results))
        }
    })
}

fn run_sims_single_winner(
    config: &Config,
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
) -> Result<(), Box<dyn Error>> {
    let (all_results, summaries) = collect_single_winner(config, trials);

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

/// Spawn the worker pool, run every trial, and fold the per-batch output back
/// together into one `Vec` of rows plus the combined per-method statistics.
fn collect_single_winner(
    config: &Config,
    trials: usize,
) -> (Vec<ExperimentResult>, Option<Vec<SendableMethodReport>>) {
    let num_workers = std::thread::available_parallelism().unwrap().get();
    let min_chunks = (num_workers * 8).max(trials.div_ceil(MAX_TRIALS_PER_JOB));
    // let min_chunks = num_workers.max(trials.div_ceil(MAX_TRIALS_PER_JOB));
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

    (all_results, summaries)
}

fn run_batch(
    config: &Config,
    trials: usize,
    task_result_tx: &mpsc::Sender<TaskResult>,
) -> Result<(), Box<dyn Error>> {
    let mut runner = TrialRunner::new(config, rand::rng());
    let results: Vec<ExperimentResult> = (0..trials).map(|_| runner.do_trial()).collect();

    task_result_tx
        .send(TaskResult {
            method_stats: runner.method_stats(),
            results,
        })
        .expect("Could not send batch results");

    Ok(())
}

pub fn get_writer(
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
    use crate::runner::tests::single_winner_config;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn run_one_batch(config: &Config, trials: usize) -> TaskResult {
        let (tx, rx) = mpsc::channel();
        run_batch(config, trials, &tx).unwrap();
        drop(tx);
        rx.recv().unwrap()
    }

    #[test]
    fn run_batch_emits_one_well_formed_result_per_trial() {
        let config = single_winner_config(None);
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
        let config = single_winner_config(Some(6));
        let result = run_one_batch(&config, 5);

        assert_eq!(result.results.len(), 5);
        for er in &result.results {
            assert_eq!(er.cand_regret.len(), 3); // 3 finalists, not 6 primary candidates
            assert_eq!(er.methods.len(), 2);
        }
    }

    #[test]
    fn run_sims_without_an_outfile_completes() {
        run_sims(&single_winner_config(None), 20, &None).unwrap();
    }

    #[test]
    fn simulate_returns_a_batch_with_one_row_per_trial() {
        let batch = simulate(&single_winner_config(None), 12).unwrap().unwrap();

        assert_eq!(batch.num_rows(), 12);
        let schema = batch.schema();
        let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(column_names.contains(&"cand_regret"));
        assert!(column_names.contains(&"methods"));
    }

    #[test]
    fn simulate_returns_none_for_zero_trials_rather_than_an_empty_batch() {
        // There are no rows to trace a schema from, so there is no batch to build.
        assert!(simulate(&single_winner_config(None), 0).unwrap().is_none());
    }

    #[test]
    fn simulate_rejects_an_invalid_config() {
        let mut config = single_winner_config(None);
        config.methods.clear();
        assert!(simulate(&config, 5).is_err());
    }

    #[test]
    fn run_sims_writes_a_parquet_that_reads_back() {
        let config = single_winner_config(None);
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
}
