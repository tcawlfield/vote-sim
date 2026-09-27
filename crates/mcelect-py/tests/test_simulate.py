"""End-to-end tests for the Python bindings.

Run against an installed wheel (`maturin develop` or `pip install -e .`), not
the source tree -- the extension module has to be built first.
"""

import json
import threading
import time

import pytest

import mcelect

SINGLE_WINNER = {
    "voters": 60,
    "candidates": 4,
    "considerations": [{"Likability": {"mean": 0.1}}],
    "methods": [{"Plurality": {"strat": "Honest"}}, {"Borda": {}}],
}

MULTI_WINNER = {
    "voters": 60,
    "candidates": 6,
    "mode": "multi_winner",
    "committee_size": 3,
    "considerations": [{"Likability": {"mean": 0.1}}],
    "committee_methods": [{"PluralityTopN": {}}],
}


def test_simulate_returns_one_row_per_trial():
    table = mcelect.simulate(SINGLE_WINNER, 50)
    assert table.num_rows == 50
    assert "cand_regret" in table.column_names
    assert set(table.column("methods").combine_chunks().type.field(i).name
               for i in range(2)) == {"Borda_h", "pl_h"}


def test_regret_is_sorted_and_starts_at_zero():
    table = mcelect.simulate(SINGLE_WINNER, 20)
    for regrets in table.column("cand_regret").to_pylist():
        assert regrets[0] == 0.0
        assert regrets == sorted(regrets)


def test_a_json_string_config_works_too():
    table = mcelect.simulate(json.dumps(SINGLE_WINNER), 10)
    assert table.num_rows == 10


def test_multi_winner_mode_elects_a_committee():
    table = mcelect.simulate(MULTI_WINNER, 25)
    assert table.num_rows == 25
    assert "ideal_cand" not in table.column_names
    for row in table.column("methods").to_pylist():
        assert len(row["pltn"]["winners"]) == MULTI_WINNER["committee_size"]


def test_invalid_config_raises_value_error():
    with pytest.raises(ValueError, match="invalid config"):
        mcelect.simulate({"voters": 10}, 5)


def test_config_failing_validation_raises_value_error():
    no_methods = SINGLE_WINNER | {"methods": []}
    with pytest.raises(ValueError, match="at least one entry"):
        mcelect.simulate(no_methods, 5)


def test_zero_trials_raises_rather_than_returning_an_empty_table():
    with pytest.raises(ValueError, match="at least 1"):
        mcelect.simulate(SINGLE_WINNER, 0)


def test_the_gil_is_released_during_a_run():
    """A Python thread must keep making progress while a simulation runs."""
    ticks = 0
    done = threading.Event()

    def spin():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        mcelect.simulate(SINGLE_WINNER | {"voters": 2000, "candidates": 8}, 4000)
    finally:
        done.set()
        spinner.join()

    assert ticks > 10, f"only {ticks} ticks; the GIL was probably held"
