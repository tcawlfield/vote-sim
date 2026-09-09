// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::sync::mpsc;

use arrow_array::RecordBatch;
use parquet::file::metadata::KeyValue;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use work_queue::Queue;

use crate::config::Config;
use crate::considerations::{ConsiderationSim, ConsiderationSimKind};
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::{MethodTracker, SendableMethodReport};
use crate::methods::Strategy;
use crate::out_types::{ExperimentResult, MethodResult};
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

pub fn run_sims(
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
            let final_candidates = rrv.multi_elect(sim_primary, None, sim.ncand);
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
        log::debug!("Cov matrix: {}", cov_matrix.elements);

        let mut method_results: BTreeMap<String, MethodResult> = BTreeMap::new();
        let mut prev_rslt = None;
        for method in methods.iter_mut() {
            let (pairing, result) = method.elect(&sim, prev_rslt);
            if let Strategy::Honest = method.method.strat() {
                prev_rslt = Some(pairing);
            }
            log::debug!(
                "Method {:?} found winner {} -- regret {}",
                method.method.name(),
                result.winner,
                result.regret
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
    for consid in axes {
        match consid.get_name().as_str() {
            "likability" => {
                likability = Some(
                    collect_positions(consid, ordered_final_cands)
                        .into_iter()
                        .map(|coords| coords[0])
                        .collect(),
                );
            }
            "issues" => issues = Some(collect_positions(consid, ordered_final_cands)),
            _ => {}
        }
    }

    ExperimentResult {
        ideal_cand: 0,
        cand_regret,
        likability,
        issues,
        cov_matrix: cov,
        num_smith: sim.smith_set_size() as u32,
        in_smith,
        methods,
    }
}

/// Collect a consideration's candidate positions as `ncand` rows of `dim`
/// coordinates, in `order`. NaN sentinels (used by considerations without a
/// spatial position) are passed through unchanged.
fn collect_positions(consid: &ConsiderationSimKind, order: &[usize]) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut current: Vec<f64> = Vec::new();
    consid.push_posn_elements(
        &mut |x, end_of_row| {
            current.push(x);
            if end_of_row {
                rows.push(std::mem::take(&mut current));
            }
        },
        order,
    );
    rows
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
