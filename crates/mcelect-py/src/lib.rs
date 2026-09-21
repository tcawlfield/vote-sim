// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for [`mcelect`].
//!
//! This is the thin half of the binding: it parses a JSON config, runs the
//! simulation with the GIL released, and hands the resulting Arrow data back as
//! a `pyarrow.Table`. Everything friendlier -- accepting a `dict`, reading TOML
//! -- lives in the pure-Python wrapper that re-exports this module.

use arrow_pyarrow::{PyArrowType, Table};
use mcelect::Config;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Run `trials` elections described by `config_json` and return the per-trial
/// results as a `pyarrow.Table`.
///
/// The simulation spawns its own worker threads, so the GIL is released for the
/// whole run.
#[pyfunction]
#[pyo3(signature = (config_json, trials))]
fn simulate(py: Python<'_>, config_json: &str, trials: usize) -> PyResult<PyArrowType<Table>> {
    if trials == 0 {
        // The Arrow schema is traced from the rows themselves, so with no rows
        // there is nothing to build a table from.
        return Err(PyValueError::new_err("trials must be at least 1"));
    }
    let config = Config::from_json_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))?;

    // `config` is owned and plain data, so nothing here touches the interpreter.
    let batch = py
        .detach(|| mcelect::simulate(&config, trials))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .ok_or_else(|| PyValueError::new_err("simulation produced no results"))?;

    let schema = batch.schema();
    Table::try_new(vec![batch], schema)
        .map(PyArrowType)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn _mcelect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
