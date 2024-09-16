use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, Int32Builder, ListBuilder};
use arrow_array::RecordBatch;
use parquet::file::metadata::KeyValue;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};

use arrow_schema::{DataType, Field, SchemaBuilder};
use std::fs;
use std::{error::Error, sync::Arc};

use crate::config::Config;
use crate::considerations::ConsiderationSim;
use crate::cov_matrix::CovMatrix;
use crate::method_tracker::MethodTracker;
use crate::methods::Strategy;
use crate::sim::Sim;

pub fn run(
    config: &Config,
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::thread_rng();

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

    // TODO: Use position arrays to demonstrate that RRV primaries spread out the candidates
    //   in position space and improve likeability.

    let mut cov_matrix = CovMatrix::new(sim.ncand);

    let mut mwms = if let Some(sim_primary) = &sim_primary {
        Some(config.primary_method.new_sim(&sim_primary))
    } else {
        None
    };

    for itrial in 0..trials {
        // println!("Sim election {}", itrial + 1);

        let final_candidates = if let Some(rrv) = &mut mwms {
            let sim_primary: &mut Sim = sim_primary.as_mut().unwrap();
            sim_primary.election(&mut axes, &mut rng, itrial == 0);
            let final_candidates = rrv.multi_elect(&sim_primary, None, sim.ncand, itrial == 0);
            if itrial == 0 {
                println!("primary election winners: {:?}", final_candidates);
            }
            sim.take_from_primary(sim_primary, &final_candidates);
            Some(final_candidates)
        } else {
            sim.election(&mut axes, &mut rng, itrial == 0);
            None
        };

        let mut prev_rslt = None;
        for method in methods.iter_mut() {
            let mut rslt = method.elect(&sim, prev_rslt, itrial == 0);
            if rslt.is_tied() {
                rslt = sim.break_tie_with_plurality(&rslt);
            }
            let regret = sim.regrets[rslt.winner.cand];
            if let Strategy::Honest = method.method.strat() {
                prev_rslt = Some(rslt);
            } else {
                // prev_rslt = None;
            }
            if itrial == 0 {
                println!(
                    "Method {:?} found winner {} -- regret {}",
                    method.method.name(),
                    rslt.winner.cand,
                    regret
                );
            }
        }

        ideal_cnd_bld.append_value(sim.cand_by_regret[0] as i32);
        for rgrt in sim.regrets.iter() {
            cand_regret_bld.values().append_value(*rgrt);
        }
        cand_regret_bld.append(true);
        cov_matrix.compute(&sim.scores);
        for ix in 0..sim.ncand {
            for iy in 0..(ix + 1) {
                cov_bld
                    .values()
                    .values()
                    .append_value(cov_matrix.elements[(ix, iy)]);
            }
            cov_bld.values().append(true); // End of row
        }
        cov_bld.append(true); // End of matrix

        for (consid, pos_bld) in axes.iter().zip(cand_posn_blds.iter_mut()) {
            consid.push_posn_elements(
                &mut |x, next_row| {
                    pos_bld.values().values().append_value(x);
                    if next_row {
                        pos_bld.values().append(true);
                    }
                },
                final_candidates,
            );
            pos_bld.append(true);
        }
    }

    let mut columns: Vec<arrow_array::ArrayRef> = Vec::new();
    columns.push(Arc::new(ideal_cnd_bld.finish()) as arrow_array::ArrayRef);
    columns.push(Arc::new(cand_regret_bld.finish()) as arrow_array::ArrayRef);
    for cpb in cand_posn_blds.iter_mut() {
        columns.push(Arc::new(cpb.finish()) as arrow_array::ArrayRef);
    }
    columns.push(Arc::new(cov_bld.finish()) as arrow_array::ArrayRef);
    for method in methods.iter_mut() {
        columns.push(method.get_column());
    }
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

    for method in methods.iter() {
        schema.push(method.get_field());
    }
    let batch = RecordBatch::try_new(Arc::new(schema.finish()), columns).unwrap();

    let config_str = serde_json::to_string(config).unwrap();
    if let Some(filename) = outfile {
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_key_value_metadata(Some(vec![KeyValue::new(
                "voting_config".to_owned(),
                config_str,
            )]))
            .build();
        let file = fs::File::create(&filename)?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();

        writer.write(&batch).expect("Writing batch");

        // writer must be closed to write footer
        writer.close().unwrap();
        println!("Wrote {}", filename.to_str().unwrap());
    }

    for method in methods.iter() {
        method.report();
    }

    Ok(())
}
