use arrow_array::builder::{Float64Builder, Int32Builder, ListBuilder};
use arrow_array::RecordBatch;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};

use arrow_schema::{DataType, Field, SchemaBuilder};
use std::fs;
use std::{error::Error, sync::Arc};

use crate::consideration::Consideration;
use crate::cov_matrix::CovMatrix;
use crate::methods::RRV;
use crate::methods::{MethodTracker, Strategy};
use crate::sim::Sim;

pub fn run(
    axes: &mut [&mut dyn Consideration],
    sim: &mut Sim,
    methods: &mut [MethodTracker],
    trials: usize,
    outfile: &Option<std::ffi::OsString>,
    sim_primary: &mut Option<Sim>,
) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::thread_rng();

    // Create Arrow array builders:
    let mut cov_bld = ListBuilder::new(ListBuilder::new(Float64Builder::new()));
    let mut ideal_cnd_bld = Int32Builder::with_capacity(trials);

    // TODO: add column(s) for candidates: ideal candidate, candidate regrets (FixedSizeList), position arrays
    // Prepend "m_" to method column names to identify these.
    // Position arrays: StructArray: likeability, p0, p1, ... (1 value per candidate)
    // TODO: Use position arrays to demonstrate that RRV primaries spread out the candidates
    //   in position space and improve likeability.

    let mut cov_matrix = CovMatrix::new(sim.ncand);

    let mut rrv = if let Some(sim_primary) = &sim_primary {
        Some(RRV::new(&sim_primary, 10, Strategy::Honest))
    } else {
        None
    };

    for itrial in 0..trials {
        // println!("Sim election {}", itrial + 1);

        if let Some(rrv) = &mut rrv {
            let sim_primary: &mut Sim = sim_primary.as_mut().unwrap();
            sim_primary.election(axes, &mut rng, itrial == 0);
            let final_candidates = rrv.multi_elect(&sim_primary, None, sim.ncand, itrial == 0);
            if itrial == 0 {
                println!("primary election winners: {:?}", final_candidates);
            }
            sim.take_from_primary(sim_primary, &final_candidates);
        } else {
            sim.election(axes, &mut rng, itrial == 0);
        }

        let mut prev_rslt = None;
        for method in methods.iter_mut() {
            let rslt = method.elect(&sim, prev_rslt, itrial == 0);
            let regret = sim.regrets[rslt.winner.cand];
            if let Strategy::Honest = method.method.strat() {
                prev_rslt = Some(rslt);
            } else {
                prev_rslt = None;
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
    }

    let mut columns: Vec<arrow_array::ArrayRef> = Vec::new();
    columns.push(Arc::new(ideal_cnd_bld.finish()) as arrow_array::ArrayRef);
    columns.push(Arc::new(cov_bld.finish()) as arrow_array::ArrayRef);
    for method in methods.iter_mut() {
        columns.push(method.get_column());
    }
    let mut schema = SchemaBuilder::new();
    schema.push(Field::new("ideal_cand", DataType::Int32, true));
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

    if let Some(filename) = outfile {
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
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
