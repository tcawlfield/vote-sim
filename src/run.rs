// © Copyright 2025 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use arrow_array::builder::{
    BooleanBuilder, FixedSizeListBuilder, Float64Builder, Int32Builder, ListBuilder,
};
use arrow_array::{RecordBatch, StructArray};
use arrow_schema::{DataType, Field, SchemaBuilder};
use parquet::file::metadata::KeyValue;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use core::task;
use std::fs;
use std::sync::mpsc;
use std::{error::Error, sync::Arc};
use work_queue::Queue;

use crate::config::Config;
use crate::considerations::ConsiderationSim;
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::{MethodTracker, SendableMethodReport};
use crate::methods::Strategy;
use crate::sim::Sim;

static MAX_TRIALS_PER_JOB: usize = 10000;

struct Task {
    config: Config,
    trials: usize,
    result_chan: mpsc::Sender<TaskResult>,
}

struct TaskResult {
    method_stats: Vec<SendableMethodReport>,
    batch: RecordBatch,
}

pub fn run(
    config: &Config,
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
) -> Result<(), Box<dyn Error>> {
    let num_workers = std::thread::available_parallelism().unwrap().get();
    let min_chunks = num_workers.max((trials + MAX_TRIALS_PER_JOB - 1) / MAX_TRIALS_PER_JOB);
    let chunks_per_worker = (min_chunks + num_workers - 1) / num_workers;
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
        let task_trials = (trials_left + chunks_to_do - 1) / chunks_to_do;
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

    // Loop over task_result_rx

    let mut writer = None;
    let mut summaries: Option<Vec<SendableMethodReport>> = None;
    while let Ok(mut task_result) = task_result_rx.recv() {
        log::info!("Completed a batch of {} elections", task_result.method_stats[0].ntrials);
        if writer.is_none() {
            if let Some(filename) = outfile {
                writer = Some(get_writer(&config, &filename, &task_result.batch));
            }
        }
        if let Some(writer) = writer.as_mut() {
            writer.write(&task_result.batch)?;
        }
        if let Some(summaries) = summaries.as_mut() {
            for (whole_summary, task_summary) in
                summaries.iter_mut().zip(task_result.method_stats.iter())
            {
                whole_summary.combine(task_summary);
            }
        } else {
            summaries = Some(std::mem::take(&mut task_result.method_stats));
        }
    }

    // for method in methods.iter() {
    //     method.report();
    // }
    if let Some(writer) = writer {
        // writer must be closed to write footer
        writer.close().unwrap();
        println!("Wrote {}", outfile.as_ref().unwrap().to_str().unwrap());
    }

    // for handle in handles {
    //     handle.join().unwrap();
    // }

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
    let ncit = config.voters;

    let mut sim = Sim::new(ncand, ncit);

    let mut sim_primary = if let Some(pcand) = config.primary_candidates {
        Some(Sim::new(pcand, ncit))
    } else {
        None
    };

    let mut axes: Vec<Box<dyn ConsiderationSim>> = {
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
        .map(|m| MethodTracker::new(m, &sim, trials))
        .collect();

    // Create Arrow array builders:
    let mut cov_bld = ListBuilder::new(ListBuilder::new(Float64Builder::new()));
    let mut ideal_cnd_bld = Int32Builder::with_capacity(trials);
    let mut cand_regret_bld = FixedSizeListBuilder::new(
        Float64Builder::with_capacity(trials * sim.ncand),
        sim.ncand as i32,
    );
    let mut cand_posn_blds = Vec::new();
    for consid in axes.iter() {
        cand_posn_blds.push(FixedSizeListBuilder::new(
            FixedSizeListBuilder::new(
                Float64Builder::with_capacity(trials * sim.ncand * consid.get_dim()),
                consid.get_dim() as i32,
            ),
            sim.ncand as i32,
        ));
    }
    let mut smith_candidates_bld = Int32Builder::with_capacity(trials);
    let mut in_smith_set_bld = FixedSizeListBuilder::new(
        BooleanBuilder::with_capacity(trials * sim.ncand),
        sim.ncand as i32,
    );

    let mut cov_matrix = CovMatrix::new(sim.ncand);

    let mut mwms = if let Some(sim_primary) = &sim_primary {
        Some(config.primary_method.new_sim(&sim_primary))
    } else {
        None
    };

    // ordered_final_cands is a list of candidates in order of increasing regret.
    // With no primary, ordered_final_cands is identical to sim.cand_by_regret.
    // With a primary, it's a list containing only winning primary candidates.
    let mut ordered_final_cands = vec![0; sim.ncand];

    for itrial in 0..trials {
        log::debug!("Sim election {}", itrial + 1);

        if let Some(rrv) = &mut mwms {
            let sim_primary: &mut Sim = sim_primary.as_mut().unwrap();
            sim_primary.election(&mut axes, &mut rng);
            let final_candidates = rrv.multi_elect(&sim_primary, None, sim.ncand);
            log::debug!("primary election winners: {:?}", final_candidates);
            sim.take_from_primary(sim_primary, &final_candidates);

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

        let mut prev_rslt = None;
        for method in methods.iter_mut() {
            let rslt = method.elect(&sim, prev_rslt);
            let regret = sim.regrets[rslt.winner.cand];
            if let Strategy::Honest = method.method.strat() {
                prev_rslt = Some(rslt);
            }
            log::debug!(
                "Method {:?} found winner {} -- regret {}",
                method.method.name(),
                rslt.winner.cand,
                regret
            );
        }

        ideal_cnd_bld.append_value(0);
        let cbr = &sim.cand_by_regret;
        for &icand in cbr.iter() {
            cand_regret_bld.values().append_value(sim.regrets[icand]);
        }
        cand_regret_bld.append(true);
        for ix in 0..sim.ncand {
            for iy in 0..(ix + 1) {
                cov_bld
                    .values()
                    .values()
                    .append_value(cov_matrix.elements[(cbr[ix], cbr[iy])]);
            }
            cov_bld.values().append(true); // End of row
        }
        cov_bld.append(true); // End of matrix

        for (consid, pos_bld) in axes.iter().zip(cand_posn_blds.iter_mut()) {
            consid.push_posn_elements(
                &mut |x, next_row| {
                    if x.is_nan() {
                        pos_bld.values().values().append_null();
                    } else {
                        pos_bld.values().values().append_value(x);
                    }
                    if next_row {
                        pos_bld.values().append(true);
                    }
                },
                &ordered_final_cands,
            );
            pos_bld.append(true);
        }
        smith_candidates_bld.append_value(sim.smith_set_size() as i32);
        for &icand in cbr.iter() {
            in_smith_set_bld
                .values()
                .append_value(sim.in_smith_set[icand]);
        }
        in_smith_set_bld.append(true);
    }

    let mut columns: Vec<arrow_array::ArrayRef> = Vec::new();
    columns.push(Arc::new(ideal_cnd_bld.finish()) as arrow_array::ArrayRef);
    columns.push(Arc::new(cand_regret_bld.finish()) as arrow_array::ArrayRef);
    for cpb in cand_posn_blds.iter_mut() {
        columns.push(Arc::new(cpb.finish()) as arrow_array::ArrayRef);
    }
    columns.push(Arc::new(cov_bld.finish()) as arrow_array::ArrayRef);
    columns.push(Arc::new(smith_candidates_bld.finish()) as arrow_array::ArrayRef);
    columns.push(Arc::new(in_smith_set_bld.finish()) as arrow_array::ArrayRef);
    let mut method_cols = Vec::new();
    for method in methods.iter_mut() {
        method_cols.push((
            Arc::new(Field::new(
                method.colname(),
                MethodTracker::data_type(),
                false,
            )),
            method.get_column(),
        ));
    }
    columns.push(Arc::new(StructArray::from(method_cols)));

    let mut schema = SchemaBuilder::new();
    schema.push(Field::new("ideal_cand", DataType::Int32, true));
    schema.push(Field::new(
        "cand_regret",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, true)),
            sim.ncand as i32,
        ),
        true,
    ));
    for consid in axes.iter() {
        schema.push(Field::new(
            consid.get_name(),
            DataType::FixedSizeList(
                Arc::new(Field::new(
                    "item",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float64, true)),
                        consid.get_dim() as i32,
                    ),
                    true,
                )),
                sim.ncand as i32,
            ),
            true,
        ));
    }
    schema.push(Field::new(
        "cov_matrix",
        DataType::List(Arc::new(Field::new(
            "item",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ))),
        true,
    ));
    schema.push(Field::new("num_smith", DataType::Int32, true));
    schema.push(Field::new(
        "in_smith",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Boolean, true)),
            sim.ncand as i32,
        ),
        true,
    ));

    //for method in methods.iter() {
    //    schema.push(method.get_field());
    //}
    let mut meth_schema_bld = SchemaBuilder::new();
    for method in methods.iter() {
        meth_schema_bld.push(Field::new(
            method.colname(),
            MethodTracker::data_type(),
            false,
        ));
    }
    schema.push(Field::new(
        "methods",
        DataType::Struct(meth_schema_bld.finish().fields),
        false,
    ));
    let batch: RecordBatch = RecordBatch::try_new(Arc::new(schema.finish()), columns).unwrap();
    let sendable_reports: Vec<SendableMethodReport> =
        methods.iter().map(|m| m.sendable_report()).collect();
    task_result_tx
        .send(TaskResult {
            method_stats: sendable_reports,
            batch,
        })
        .expect("Could not send batch summry");

    Ok(())
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
    let file = fs::File::create(&filename).unwrap();
    let writer = ArrowWriter::try_new(file, sample_batch.schema(), Some(props)).unwrap();
    writer
}
