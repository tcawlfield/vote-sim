# mcelect (Python bindings)

Run [mcelect](../mcelect) election simulations from Python and get the results
back as a `pyarrow.Table`, skipping the parquet round-trip.

```python
import mcelect

config = mcelect.load_config("configs/default.toml")
table = mcelect.simulate(config, 100_000)
```

`config` is a plain dict (or a JSON string) with the same shape as the TOML
config files the `mcelect` command-line tool reads, so `load_config` on a `.toml`
file and passing the result straight through is the usual path. The columns
depend on the config's `mode`: `ExperimentResult`'s fields for `single_winner`,
`CommitteeResult`'s for `multi_winner`.

The simulation spawns one worker thread per core and releases the GIL for the
whole run, so other Python threads keep going while it works.

Awkward Array reads the result directly, which is convenient given the nested
columns:

```python
import awkward as ak

arr = ak.from_arrow(table)
ak.mean(arr.methods.pl_h.regret)
```

Note that unlike the command-line tool, this reports no summary statistics --
every trial comes back as a row, so compute what you need from the table.

## Building

```sh
pip install maturin
cd crates/mcelect-py
maturin develop --release     # into the active virtualenv
maturin build --release       # or a wheel in ../../target/wheels
```

The wheel is abi3 for Python ≥ 3.12, so one build covers every newer
interpreter.

`mcelect-py` is deliberately excluded from the workspace's `default-members`: it
is a `cdylib` that links against libpython, which a bare `cargo test` at the
workspace root has no business building. Use `-p mcelect-py` to reach it.

## Tests

`tests/` is a pytest suite that runs against the installed extension module:

```sh
maturin develop && pytest tests -q
```
