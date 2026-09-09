//! Plain-data structs for the per-trial output of the simulation.
//!
//! The simulation loop fills a `Vec<ExperimentResult>` (one entry per election
//! trial); [`ExperimentResult::to_record_batch`] serializes a slice of them into
//! the Arrow `RecordBatch` that gets written to parquet, via `serde_arrow`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{FieldRef, Schema};
use serde_arrow::schema::{SchemaLike, TracingOptions};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ExperimentResult {
    /// Regret-rank of the utilitarian-ideal candidate. Always 0: candidates are
    /// reported in increasing-regret order, so the ideal one is first.
    pub ideal_cand: u32,
    /// Per-candidate regret, in increasing-regret order (length `ncand`).
    pub cand_regret: Vec<f64>,
    /// Candidate likability scores in the same order, when the config has a
    /// Likability consideration (length `ncand`).
    pub likability: Option<Vec<f64>>,
    /// Candidate positions in issue space in the same order, when the config has
    /// an Issues consideration (`ncand` rows of `dim` coordinates).
    pub issues: Option<Vec<Vec<f64>>>,
    /// Lower-triangular candidate/candidate utility covariance, reordered by
    /// increasing regret (row `i` has `i + 1` entries).
    pub cov_matrix: Vec<Vec<f64>>,
    /// Number of candidates in the Smith set.
    pub num_smith: u32,
    /// Whether each candidate (increasing-regret order) is in the Smith set.
    pub in_smith: Vec<bool>,
    /// One entry per voting method, keyed by the method's column name. Becomes an
    /// Arrow struct column with one field per method (BTreeMap => stable field
    /// order); the field set is discovered from the data, not the type.
    pub methods: BTreeMap<String, MethodResult>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct MethodResult {
    /// Regret-rank of the method's winner (0 == utilitarian-ideal winner).
    pub winner: u32,
    /// Regret of the method's winner.
    pub regret: f64,
}

impl ExperimentResult {
    /// The Arrow fields for the output columns, inferred from a batch of results.
    ///
    /// Uses the data (not just the type) so that `methods` becomes a struct with
    /// one field per method and the optional consideration columns take the inner
    /// type from whichever results carry them.
    pub fn arrow_fields(results: &[ExperimentResult]) -> Vec<FieldRef> {
        Vec::<FieldRef>::from_samples(results, tracing_options())
            .expect("ExperimentResult maps to an Arrow schema")
    }

    /// Serialize a slice of trial results into one Arrow `RecordBatch`.
    pub fn to_record_batch(results: &[ExperimentResult]) -> RecordBatch {
        let fields = Self::arrow_fields(results);
        let columns =
            serde_arrow::to_arrow(&fields, results).expect("trial results serialize to Arrow");
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }
}

fn tracing_options() -> TracingOptions {
    TracingOptions::default()
        // Trace `methods` (a BTreeMap) as a struct keyed by method column name.
        .map_as_struct(true)
        // An absent consideration column serializes as all-null.
        .allow_null_fields(true)
        // Prefer the conventional 32-bit-offset List / Utf8 over the "large" forms.
        .sequence_as_large_list(false)
        .strings_as_large_utf8(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_array::cast::AsArray;

    fn method(winner: u32, regret: f64) -> MethodResult {
        MethodResult { winner, regret }
    }

    fn sample(num_smith: u32, with_issues: bool) -> ExperimentResult {
        ExperimentResult {
            ideal_cand: 0,
            cand_regret: vec![0.0, 1.0, 2.5],
            likability: Some(vec![0.3, 0.1, 0.2]),
            issues: with_issues.then(|| vec![vec![0.0, 1.0], vec![-1.0, 0.5], vec![2.0, -2.0]]),
            cov_matrix: vec![vec![1.0], vec![0.2, 1.5], vec![-0.1, 0.3, 2.0]],
            num_smith,
            in_smith: vec![true, false, false],
            methods: BTreeMap::from([
                ("pl_h".to_string(), method(0, 0.0)),
                ("irv_h".to_string(), method(2, 2.5)),
            ]),
        }
    }

    #[test]
    fn to_record_batch_has_a_column_per_field_and_a_row_per_result() {
        let results = [sample(1, true), sample(3, true)];
        let batch = ExperimentResult::to_record_batch(&results);

        assert_eq!(batch.num_rows(), 2);
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            [
                "ideal_cand",
                "cand_regret",
                "likability",
                "issues",
                "cov_matrix",
                "num_smith",
                "in_smith",
                "methods",
            ]
        );

        let num_smith = batch
            .column_by_name("num_smith")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt32Type>();
        assert_eq!(num_smith.values(), &[1, 3]);
    }

    #[test]
    fn methods_becomes_a_struct_keyed_by_method_name() {
        let batch = ExperimentResult::to_record_batch(&[sample(1, true)]);
        let methods = batch.column_by_name("methods").unwrap().as_struct();
        // BTreeMap key order.
        let fields: Vec<&str> = methods.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(fields, ["irv_h", "pl_h"]);

        let irv = methods.column_by_name("irv_h").unwrap().as_struct();
        let winner = irv
            .column_by_name("winner")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt32Type>();
        assert_eq!(winner.values(), &[2]);
    }

    #[test]
    fn absent_consideration_still_produces_a_readable_column() {
        // No result carries `issues`; the column must still exist, one entry per
        // row, and hold no data (serde_arrow traces it as the Null type).
        let results = [sample(1, false), sample(2, false)];
        let batch = ExperimentResult::to_record_batch(&results);
        let issues = batch.column_by_name("issues").unwrap();
        assert_eq!(issues.len(), 2);
        assert_eq!(issues.data_type(), &arrow_schema::DataType::Null);
    }
}
