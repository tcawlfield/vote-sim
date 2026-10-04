# CLAUDE.md

Monte-Carlo voting-method simulator. Rust core, CLI, and Python bindings. See
README.md for concepts (considerations, utilities, VSE) and NOTES.md for naming
and the to-do list.

## Layout

- `crates/mcelect` — the library: config, considerations (utility generators),
  voting methods, runner, Arrow/Parquet output.
- `crates/mcelect-cli` — thin clap wrapper; reads a TOML/YAML config, writes
  Parquet.
- `crates/mcelect-py` — PyO3/maturin bindings (`mcelect.simulate`,
  `load_config`) plus pydantic config models in `python/mcelect/config.py`.
- `configs/` — example configs. The Python tests round-trip every one of them,
  so a new or changed config must stay valid.
- `results/`, `study/` — analysis notebooks; not part of the build.

## Commands

    cargo fmt --all -- --check # pre-commit hook cargo clippy --all-targets
    --all-features -- -D warnings cargo test --all-features # these three =
    pre-push hook RUSTDOCFLAGS=-D\ warnings cargo doc --no-deps --all-features #
    CI also checks this

    # Python bindings (not in default-members; always use -p or maturin) cargo
    clippy -p mcelect-py --all-targets --features pyo3/extension-module -- -D
    warnings cd crates/mcelect-py && maturin develop --release && pytest tests
    -q

    cargo run --release -p mcelect-cli -- -c configs/default.toml -t 1000 -o
    out.parquet

Hooks live in `.githooks/` (`git config core.hooksPath .githooks`). Never use
`--no-verify`. The user's shell is fish.

## Keeping Rust and Python in sync

`mcelect.config` hand-mirrors the Rust config structs. When you add or change a
method, consideration, or config field:

1. Rust: the struct, plus every match arm in `methods/mod.rs` (`Method` /
   multi-winner enum: `new_sim`, `colname`, …).
2. Python: the pydantic model in `config.py`, its entry in the `_tagged_union`,
   and `__all__`.
3. Defaults live **only** in Rust. Python fields default to `None` and are
   omitted from the JSON, so don't duplicate a default in Python; mention it in
   the field doc instead.
4. Serde aliases on the Rust side should be accepted by the Python model too.
5. Run the pytest suite; `test_config.py` catches most drift.

## Conventions

- Every source file starts with the copyright +
  `SPDX-License-Identifier: Apache-2.0` header.
- Say "voter", not "citizen" (`nvtr` / `ivtr` for counts/indices); "candidate"
  rather than "alternative".
- Enums are externally tagged in config (`{"Plurality": {"strat": "Honest"}}`);
  unknown fields are errors in both Rust and Python.
- A method's output column name (`colname`) is its identity; duplicates are
  rejected. `PartialEq` on `Method` means "elects the same way" (used to find a
  strategic method's honest pre-poll).
- Test names read as sentences
  (`test_unset_fields_are_left_for_rust_to_default`). Method unit tests share
  helpers in `methods/test_utils.rs`.
- Sanity cross-check: with a single 1-D issue consideration there should be no
  Condorcet cycles.

## Git

Work on `feature/<name>` branches and open PRs against `main` (CI runs on
`main`).
