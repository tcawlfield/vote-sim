"""Tests for `mcelect.config`, the pydantic mirror of the Rust config structs.

The Rust side is the reference throughout: `normalize_config` parses a config
with the Rust definitions and re-serializes it with every default filled in.
"""

import json
from pathlib import Path

import pydantic
import pytest

import mcelect
from mcelect._mcelect import normalize_config
from mcelect.config import (
    RRV,
    STAR,
    Borda,
    BtrIrv,
    Config,
    Electorate,
    Faction,
    InstantRunoff,
    Irrational,
    Issue,
    Issues,
    Likability,
    Minimax,
    Multivote,
    Plurality,
    PluralityTopN,
    QGaussian2,
    RangeVoting,
    RP,
)

CONFIGS = sorted((Path(__file__).parents[3] / "configs").glob("*.toml"))


def rust_view(config_json: str) -> dict:
    return json.loads(normalize_config(config_json))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_repo_config_means_the_same_after_a_trip_through_python(path):
    """Loading a TOML config into the models and dumping it again must give Rust
    exactly the config it reads from the file itself."""
    original = mcelect.load_config(path)
    via_python = Config.from_file(path).to_json()
    assert rust_view(via_python) == rust_view(json.dumps(original))


def every_variant_config() -> Config:
    """A config using every model, so nothing goes unexercised."""
    return Config(
        voters=100,
        candidates=4,
        primary_candidates=6,
        primary_method=RRV(strat="Honest", ranks=10, k=0.7),
        considerations=[
            Likability(mean=0.5),
            Issues([Issue(sigma=1.0, halfcsep=0.5, sigma_vtr=0.8, halfvsep=0.2,
                          uniform=True, horizon=3.0)]),
            Irrational(sigma=0.3, camps=2, individualism_deg=30.0),
            Electorate(
                dimensions=2,
                distance_function=QGaussian2(1.5),
                factions=[
                    Faction(popularity=0.6, voter_center=[-1.0, 0.0], voter_spread=0.5,
                            candidate_center=[-0.8, 0.1], candidate_spread=0.3,
                            in_group_likability_ceiling=0.2),
                    Faction(popularity=0.4, voter_center=[1.0, 0.0], voter_spread=0.5),
                ],
            ),
        ],
        methods=[
            Plurality(strat="Honest"),
            Plurality(strat="Strategic"),
            RangeVoting(strat="Strategic", nranks=10, strategic_stretch_factor=2.0,
                        colname="range_s_2x"),
            InstantRunoff(),
            Borda(strat="Honest", rank_top_n=3),
            Multivote(strat="Honest", votes=3, spread_fact=1.0),
            STAR(strat="Strategic", nranks=6, strategic_stretch_factor=2.0),
            RP(strat="Honest"),
            BtrIrv(),
            Minimax(colname="minimax"),
        ],
    )


def test_every_model_is_accepted_by_rust_with_its_values_intact():
    config = every_variant_config()
    rust = rust_view(config.to_json())

    assert rust["primary_method"] == {"RRV": {"strat": "Honest", "ranks": 10, "k": 0.7}}
    likability, issues, irrational, electorate = rust["considerations"]
    assert likability == {"Likability": {"mean": 0.5}}
    assert issues["Issues"][0]["horizon"] == 3.0
    assert irrational == {"Irrational": {"sigma": 0.3, "camps": 2, "individualism_deg": 30.0}}
    assert electorate["Electorate"]["distance_function"] == {"QGaussian2": 1.5}
    assert electorate["Electorate"]["factions"][0]["candidate_center"] == [-0.8, 0.1]
    assert rust["methods"][2]["Range"] == {
        "strat": "Strategic", "nranks": 10, "strategic_stretch_factor": 2.0,
        "colname": "range_s_2x",
    }
    assert rust["methods"][9] == {"MM": {"colname": "minimax"}}


def test_every_model_round_trips_through_its_own_json():
    config = every_variant_config()
    assert Config.model_validate_json(config.to_json()) == config


def test_unset_fields_are_left_for_rust_to_default():
    config = Config(
        voters=10,
        candidates=3,
        considerations=[Issues([Issue(sigma=1.0, halfcsep=0.0)])],
        methods=[STAR(strat="Honest"), Borda()],
    )
    assert json.loads(config.to_json())["methods"] == [
        {"STAR": {"strat": "Honest"}},
        {"Borda": {}},
    ]
    rust = rust_view(config.to_json())
    assert rust["mode"] == "single_winner"
    assert rust["methods"][0]["STAR"]["nranks"] == 6
    assert rust["methods"][1]["Borda"]["strat"] == "Honest"
    assert rust["considerations"][0]["Issues"][0]["uniform"] is False


def test_the_tagged_dict_form_validates_too():
    config = Config.model_validate({
        "voters": 10,
        "candidates": 3,
        "considerations": [{"Likability": {"mean": 0.1}}],
        "methods": [{"Plurality": {"strat": "Honest"}}, {"Range": {"strat": "Honest", "nranks": 5}}],
    })
    assert config.methods == [Plurality(strat="Honest"), RangeVoting(strat="Honest", nranks=5)]


def test_rust_aliases_are_accepted_and_written_canonically():
    config = Config.model_validate({
        "voters": 10,
        "candidates": 3,
        "considerations": [
            {"likability": {"mean": 0.1}},
            {"electorate": {
                "dimensions": 1,
                "distance_function": {"q_gaussian_2": 1.0},
                "factions": [{"popularity": 1.0, "voter_center": [0.0], "voter_spread": 1.0}],
            }},
        ],
        "methods": [{"Plurality": {"strat": "Honest"}}],
    })
    dumped = json.loads(config.to_json())
    assert list(dumped["considerations"][0]) == ["Likability"]
    assert dumped["considerations"][1]["Electorate"]["distance_function"] == {"QGaussian2": 1.0}

    euclid = Electorate.model_validate({
        "dimensions": 1, "distance_function": "negative_euclidean",
        "factions": [{"popularity": 1.0, "voter_center": [0.0], "voter_spread": 1.0}],
    })
    assert euclid.distance_function == "NegativeEuclidean"


def test_unknown_fields_are_rejected_like_in_rust():
    with pytest.raises(pydantic.ValidationError, match="strategic_strech_factor"):
        RangeVoting(strat="Strategic", nranks=10, strategic_strech_factor=2.0)
    with pytest.raises(pydantic.ValidationError, match="colnmae"):
        Config.model_validate({
            "voters": 10, "candidates": 3, "considerations": [],
            "methods": [{"Plurality": {"strat": "Honest", "colnmae": "x"}}],
        })


def test_an_unknown_method_names_the_valid_ones():
    with pytest.raises(pydantic.ValidationError, match="expected a single key naming one of: Plurality, Range"):
        Config.model_validate({
            "voters": 10, "candidates": 3, "considerations": [],
            "methods": [{"Plurlity": {"strat": "Honest"}}],
        })


def test_a_bad_strategy_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        Plurality(strat="Sneaky")


def test_simulate_accepts_a_config_model():
    config = Config(
        voters=50,
        candidates=4,
        considerations=[Likability(mean=0.1)],
        methods=[Plurality(strat="Strategic", colname="pl_tactical")],
    )
    table = mcelect.simulate(config, 10)
    assert table.num_rows == 10
    assert [f.name for f in table.schema.field("methods").type] == ["pl_tactical"]


def test_normalized_shows_the_rust_view_and_runs_rust_checks():
    config = Config(
        voters=10,
        candidates=3,
        considerations=[Likability(mean=0.1)],
        mode="multi_winner",
        committee_size=2,
        committee_methods=[PluralityTopN()],
    )
    assert config.normalized()["committee_methods"] == [{"PluralityTopN": {}}]

    # Rust's cross-field checks, which the models don't duplicate.
    config.committee_methods = [PluralityTopN(), PluralityTopN()]
    with pytest.raises(ValueError, match="would both write output column `pltn`"):
        config.normalized()
