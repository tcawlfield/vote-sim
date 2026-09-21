// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::sync::mpsc;

use work_queue::Queue;

use crate::config::Config;
use crate::method_tracker::SendableMethodReport;
use crate::out_types::CommitteeResult;
use crate::run::get_writer;
use crate::runner::Runner;

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
    let mut runner = Runner::new(config, rand::rng());
    let results: Vec<CommitteeResult> = (0..trials).map(|_| runner.do_committee_trial()).collect();

    task_result_tx
        .send(CommitteeTaskResult {
            method_stats: runner.method_stats(),
            results,
        })
        .expect("Could not send batch results");

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::run_sims;
    use crate::runner::tests::multi_winner_config;

    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn run_one_committee_batch(config: &Config, trials: usize) -> CommitteeTaskResult {
        let (tx, rx) = mpsc::channel();
        run_batch_multi_winner(config, trials, &tx).unwrap();
        drop(tx);
        rx.recv().unwrap()
    }

    #[test]
    fn run_batch_multi_winner_emits_one_well_formed_committee_per_trial() {
        let config = multi_winner_config();
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
        let config = multi_winner_config();
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
        let mut config = multi_winner_config();
        config.committee_size = None;
        assert!(run_sims(&config, 5, &None).is_err());
    }
}
