# CLAUDE.md

Monte-Carlo voting-method simulator. Rust core, CLI, and Python bindings. See README.md
for concepts (considerations, utilities, VSE) and NOTES.md for naming and the to-do list.

## Layout

- `crates/mcelect` — the library: config, considerations (utility generators), voting
  methods, runner, Arrow/Parquet output.
- `crates/mcelect-cli` — thin clap wrapper; reads a TOML/YAML config, writes Parquet.
- `crates/mcelect-py` — PyO3/maturin bindings (`mcelect.simulate`, `load_config`) plus
  pydantic config models in `python/mcelect/config.py`.
- `configs/` — example configs. The Python tests round-trip every one of them, so a new or
  changed config must stay valid.
- `results/`, `study/` — analysis notebooks; not part of the build.

## Architecture notes

- Single-winner methods implement `MethodSim::elect(sim, poll) -> &Standings`: every
  candidate exactly once, winner first. Scores are method-specific; only the order is
  shared. Add every new method to `every_method()` in `methods/mod.rs`'s tests; the
  standings property test and the committee adapters' tests iterate over it.
- Committee methods implement `MWMethodSim`. Any single-winner method becomes one through
  `TopStandings` (first k of one election) or `Sequential` (a fresh election per seat,
  on a reduced `Sim` built with `Sim::take_from_primary`); both use `Standalone`.
- `committee_welfare.rs`: W̄ has a closed form (`rank_weights`). W* is found by top-k
  (equal weights), branch and bound in `OwaSearch` (non-increasing weights only; the
  bound relies on submodularity), or an exhaustive search. The `Harmonic` voting method
  reuses `OwaSearch` on ballots.
- Sims allocate their buffers once, in `new_sim` or on the first trial when the size
  depends on the committee, and reuse them. This is a convention rather than a measured
  need, but keep buffers with the sim that uses them.

## Commands

```sh
cargo fmt --all -- --check                                # pre-commit hook
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features                                 # these three = pre-push hook
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features   # CI also checks this

# Python bindings (not in default-members; always use -p or maturin)
cargo clippy -p mcelect-py --all-targets --features pyo3/extension-module -- -D warnings
cd crates/mcelect-py
uv sync --extra test --reinstall-package mcelect   # rebuild the extension after Rust changes
uv run --extra test pytest tests -q

cargo run --release -p mcelect-cli -- -c configs/default.toml -t 1000 -o out.parquet
cargo bench -p mcelect --bench committee_welfare   # committee welfare searches vs a trial
```

`cargo run` rebuilds first; running `target/release/mcelect` directly can use a stale
binary.

uv caches the built extension: after changing Rust code, `--reinstall-package mcelect` is
needed or the tests run against the stale build. The Python sources are installed
editable.

Hooks live in `.githooks/` (`git config core.hooksPath .githooks`). Never use
`--no-verify`. The user's shell is fish.

## Keeping Rust and Python in sync

`mcelect.config` hand-mirrors the Rust config structs. When you add or change a method,
consideration, or config field:

1. Rust: the struct, plus every match arm in `methods/mod.rs` (`Method` / multi-winner
   enum: `new_sim`, `colname`, …).
2. Python: the pydantic model in `config.py`, its entry in the `_tagged_union`, and
   `__all__`.
3. Defaults live **only** in Rust. Python fields default to `None` and are omitted from
   the JSON, so don't duplicate a default in Python; mention it in the field doc instead.
4. Serde aliases on the Rust side should be accepted by the Python model too.
5. Run the pytest suite. `test_schema_sync.py` compares the pydantic models against the
   Rust JSON Schema (schemars, behind `mcelect`'s `schema` feature): fields, variants and
   tags, required-vs-defaulted, and types. So every config type needs
   `#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]`. `test_config.py`
   covers what the schema can't: aliases, round-trips, and checks that span several
   fields.

## Conventions

- Every source file starts with the copyright + `SPDX-License-Identifier: Apache-2.0`
  header.
- Say "voter", not "citizen" (`nvtr` / `ivtr` for counts/indices); "candidate" rather than
  "alternative".
- Enums are externally tagged in config (`{"Plurality": {"strat": "Honest"}}`); unknown
  fields are errors in both Rust and Python.
- A method's output column name (`colname`) is its identity; duplicates are rejected.
  `PartialEq` on `Method` means "elects the same way" (used to find a strategic method's
  honest pre-poll).
- Test names read as sentences (`test_unset_fields_are_left_for_rust_to_default`). Method
  unit tests share helpers in `methods/test_utils.rs`.
- Sanity cross-check: with a single 1-D issue consideration there should be no Condorcet
  cycles.
- Randomness: elections draw from the runner's RNG, and methods must never consume it
  (`ties_never_change_the_elections`). Tally-based methods settle ties for first with a
  private `tie_rng` seeded from their column name (`tallies::tie_rng`). No
  `rand::random()` or global RNG in library code.
- Validation: parameter constraints go in the struct's `validate()`, which
  `Config::validate` calls and reports with the entry's position
  ("`methods` entry 2: Range needs nranks >= 2, not 1"). Test the exact message.
- Default column names include only integer parameters; floating-point ones (RRV's and
  Harmonic's `k`, STAR's stretch factor) are left out, and users set `colname` to tell
  such variants apart.
- The bindings return plain Arrow and stay neutral about analysis tooling (Awkward,
  Polars, DuckDB). Analysis measures such as the Gallagher index belong in a separate
  analysis library, not in this crate.
- A refactor meant to leave results unchanged should be checked by diffing seeded
  `TrialRunner` output over `configs/` before and after.

## Git

Work on `feature/<name>` branches and open PRs against `main` (CI runs on `main`).
