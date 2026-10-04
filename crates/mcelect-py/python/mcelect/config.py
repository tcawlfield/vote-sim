"""Pydantic models mirroring the Rust `mcelect::Config` and everything in it.

Build a config in Python and hand it to `mcelect.simulate`, or serialize it with
`Config.to_json()`::

    from mcelect.config import Config, Issue, Issues, Likability, Plurality, RangeVoting

    config = Config(
        voters=1000,
        candidates=5,
        considerations=[Likability(mean=0.5), Issues([Issue(sigma=1.0, halfcsep=0.5)])],
        methods=[
            Plurality(strat="Honest"),
            RangeVoting(strat="Strategic", nranks=10, colname="range_s"),
        ],
    )
    table = mcelect.simulate(config, 10_000)

Existing TOML and JSON configs load too, with `Config.from_file(path)` or
`Config.model_validate(mcelect.load_config(path))`.

Conventions, all following from how the Rust side reads its config:

* Enums are *externally tagged*: a method serializes as
  ``{"Plurality": {"strat": "Honest"}}``, and each such variant is a model here
  whose serialized form carries that tag. The tagged form is also accepted
  when validating.
* A field left as ``None`` is omitted from the output, so Rust applies its own
  default -- the defaults live in one place only. `to_json()` does this.
* Unknown fields are errors, as they are in Rust.

The Rust side still has the last word: `Config.normalized()` shows a config as
Rust sees it, and its checks that span several fields (a mode's required
fields, unique column names) run there.
"""

from __future__ import annotations

import json
import os
import tomllib
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    NonNegativeInt,
    RootModel,
    Tag,
    model_serializer,
    model_validator,
)

__all__ = [
    "Config",
    "RunMode",
    "Strategy",
    # Considerations
    "Consideration",
    "Likability",
    "Issues",
    "Issue",
    "Irrational",
    "Electorate",
    "Faction",
    "DistanceFunction",
    "QGaussian2",
    # Single-winner methods
    "Method",
    "Plurality",
    "RangeVoting",
    "InstantRunoff",
    "Borda",
    "Multivote",
    "STAR",
    "RP",
    "BtrIrv",
    "Minimax",
    # Multi-winner methods
    "MultiWinMethod",
    "RRV",
    "PluralityTopN",
]

Strategy = Literal["Honest", "Strategic"]
RunMode = Literal["single_winner", "multi_winner"]


class _Strict(BaseModel):
    """A plain Rust struct: unknown fields are rejected, as by serde's
    `deny_unknown_fields`."""

    model_config = ConfigDict(extra="forbid")


def _untag(cls: Any, data: Any) -> Any:
    """Unwrap the tagged form ``{TAG: payload}`` (or an alias of TAG) to the payload."""
    if isinstance(data, dict) and len(data) == 1:
        ((key, value),) = data.items()
        if key == cls.TAG or key in cls.ALIASES:
            return value
    return data


class _Variant(_Strict):
    """A struct-valued variant of an externally tagged Rust enum, serialized as
    ``{TAG: {fields}}``."""

    TAG: ClassVar[str]
    ALIASES: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _accept_tagged(cls, data: Any) -> Any:
        return _untag(cls, data)

    @model_serializer(mode="wrap")
    def _serialize_tagged(self, handler: Any) -> dict[str, Any]:
        return {self.TAG: handler(self)}


RootT = TypeVar("RootT")


class _RootVariant(RootModel[RootT], Generic[RootT]):
    """A variant of an externally tagged Rust enum whose payload isn't a struct
    (a list, or a bare number), serialized as ``{TAG: payload}``."""

    TAG: ClassVar[str]
    ALIASES: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _accept_tagged(cls, data: Any) -> Any:
        return _untag(cls, data)

    @model_serializer(mode="wrap")
    def _serialize_tagged(self, handler: Any) -> dict[str, Any]:
        return {self.TAG: handler(self)}


def _tagged_union(*variants: type) -> Any:
    """The union of an enum's variants, told apart by tag -- a model instance by
    its class, a dict by its single key (canonical name or alias)."""
    by_name = {}
    for variant in variants:
        for name in (variant.TAG, *variant.ALIASES):
            by_name[name] = variant.TAG

    def tag_of(value: Any) -> str | None:
        if isinstance(value, dict):
            return by_name.get(next(iter(value))) if len(value) == 1 else None
        return getattr(value, "TAG", None)

    names = ", ".join(v.TAG for v in variants)
    return Annotated[
        Union[tuple(Annotated[v, Tag(v.TAG)] for v in variants)],
        Discriminator(
            tag_of,
            custom_error_type="unknown_variant",
            custom_error_message=f"expected a single key naming one of: {names}",
        ),
    ]


# --- Considerations ---------------------------------------------------------


class Likability(_Variant):
    """Each candidate has a universally appealing charisma."""

    TAG: ClassVar[str] = "Likability"
    ALIASES: ClassVar[tuple[str, ...]] = ("likability",)

    mean: float
    """The scale of the Likability scores."""


class Issue(_Strict):
    """One axis of issue space."""

    sigma: float
    """The scale of the issue."""
    sigma_vtr: float | None = None
    """A separate sigma for voters, if given."""
    halfcsep: float
    """Polarization gap for candidates: half are shifted by -halfcsep, half by +halfcsep."""
    halfvsep: float | None = None
    """Polarization gap for voters, like halfcsep."""
    uniform: bool | None = None
    """Draw positions from a uniform distribution (same sigma) instead of normal.
    Rust default: false."""
    horizon: float | None = None
    """Maximum voter-candidate separation on this axis that a voter cares about.
    Rust default: effectively unlimited."""


class Issues(_RootVariant[list[Issue]]):
    """Positions in an issue space, one `Issue` per axis: ``Issues([Issue(...), ...])``."""

    TAG: ClassVar[str] = "Issues"
    ALIASES: ClassVar[tuple[str, ...]] = ("issues",)


class Irrational(_Variant):
    """Random utilities, optionally shared within voter camps."""

    TAG: ClassVar[str] = "Irrational"
    ALIASES: ClassVar[tuple[str, ...]] = ("irrational",)

    sigma: float
    """Standard deviation of the uniformly-distributed scores."""
    camps: NonNegativeInt
    """Voters fall into camps when camps > 1 (camp index = voter index % camps)."""
    individualism_deg: float
    """How far (0-90 degrees) individuals deviate from their camp. 0 means not at all."""


class QGaussian2(_RootVariant[float]):
    """Perceived utility is 1/(1 + x^2 / sigma^2), x being the voter-candidate
    distance: ``QGaussian2(sigma)``."""

    TAG: ClassVar[str] = "QGaussian2"
    ALIASES: ClassVar[tuple[str, ...]] = ("q_gaussian_2",)


def _canonical_distance_name(value: Any) -> Any:
    return "NegativeEuclidean" if value == "negative_euclidean" else value


DistanceFunction = Annotated[
    Union[Literal["NegativeEuclidean"], QGaussian2],
    BeforeValidator(_canonical_distance_name),
]
"""How a voter-candidate distance becomes a perceived utility: the string
``"NegativeEuclidean"`` (negative Euclidean distance) or a `QGaussian2`."""


class Faction(_Strict):
    """One faction of an `Electorate`."""

    popularity: float
    """Relative weight for assigning voters to this faction."""
    voter_center: list[float]
    """Center of this faction's voters in issue space."""
    candidate_center: list[float] | None = None
    """Center of this faction's candidates. Rust default: `voter_center`."""
    voter_spread: float
    """Std-dev of the scatter of voters around the center."""
    candidate_spread: float | None = None
    """Scatter of candidates around their center. Rust default: `voter_spread`."""
    in_group_likability_ceiling: float | None = None
    """Upper bound of the likability bonus seen only by voters in the candidate's
    own faction. Rust default: 0."""


class Electorate(_Variant):
    """Voters and candidates grouped into factions in an issue space."""

    TAG: ClassVar[str] = "Electorate"
    ALIASES: ClassVar[tuple[str, ...]] = ("electorate",)

    dimensions: NonNegativeInt
    """Dimensionality of the issue space. Every `*_center` must be this long."""
    distance_function: DistanceFunction | None = None
    """Rust default: "NegativeEuclidean"."""
    factions: list[Faction]


Consideration = _tagged_union(Likability, Issues, Irrational, Electorate)
"""One of the ways voters' utilities for candidates are generated."""


# --- Methods ----------------------------------------------------------------
# Every method takes an optional `colname`, replacing its default output column
# name.


class Plurality(_Variant):
    """Plurality, of First-past-the-post voting."""

    TAG: ClassVar[str] = "Plurality"

    strat: Strategy
    colname: str | None = None


class RangeVoting(_Variant):
    """Range (score) voting. With ``nranks=2`` this is approval voting."""

    TAG: ClassVar[str] = "Range"

    strat: Strategy
    nranks: int
    strategic_stretch_factor: float | None = None
    """Rust default: 4.0. Affects strategic ballots only."""
    colname: str | None = None


class InstantRunoff(_Variant):
    """Instant-runoff voting. Also known as "ranked-choice voting"."""

    TAG: ClassVar[str] = "InstantRunoff"

    colname: str | None = None


class Borda(_Variant):
    """Borda count voting."""

    TAG: ClassVar[str] = "Borda"

    strat: Strategy | None = None
    """Rust default: "Honest"."""
    rank_top_n: NonNegativeInt | None = None
    """Rank only each voter's top N candidates. Rust default: rank all."""
    colname: str | None = None


class Multivote(_Variant):
    """Multivote, where each voter can cast multiple votes."""

    TAG: ClassVar[str] = "Multivote"

    strat: Strategy
    votes: int
    spread_fact: float
    colname: str | None = None


class STAR(_Variant):
    """Score then run (STAR) voting."""

    TAG: ClassVar[str] = "STAR"

    strat: Strategy
    nranks: int | None = None
    """Rust default: 6."""
    strategic_stretch_factor: float | None = None
    """Rust default: 4.0. Affects strategic ballots only."""
    colname: str | None = None


class RP(_Variant):
    """Ranked pairs."""

    TAG: ClassVar[str] = "RP"

    strat: Strategy
    colname: str | None = None


class BtrIrv(_Variant):
    """Bottom-two-runoff instant-runoff voting."""

    TAG: ClassVar[str] = "BtrIrv"

    colname: str | None = None


class Minimax(_Variant):
    """Minimax voting."""

    TAG: ClassVar[str] = "MM"

    colname: str | None = None


Method = _tagged_union(
    Plurality, RangeVoting, InstantRunoff, Borda, Multivote, STAR, RP, BtrIrv, Minimax
)
"""A single-winner voting method."""


class RRV(_Variant):
    """Reweighted range voting."""

    TAG: ClassVar[str] = "RRV"

    strat: Strategy
    ranks: int
    k: float
    colname: str | None = None


class PluralityTopN(_Variant):
    """Elects the top N plurality vote-getters."""

    TAG: ClassVar[str] = "PluralityTopN"

    colname: str | None = None


MultiWinMethod = _tagged_union(RRV, PluralityTopN)
"""A multi-winner voting method."""


# --- The config itself ------------------------------------------------------


class Config(_Strict):
    """A complete simulation config, mirroring the Rust `mcelect::Config`."""

    voters: NonNegativeInt
    """Number of voters."""
    candidates: NonNegativeInt
    """Number of candidates."""
    primary_candidates: NonNegativeInt | None = None
    """Draw finalists from a larger field of this many via `primary_method`."""
    considerations: list[Consideration]
    """List of considerations for the simulation, summed together to produce
    the table of perceived utilities by each voter, for each candidate."""
    mode: RunMode | None = None
    """Rust default: "single_winner"."""
    methods: list[Method] = Field(default_factory=list)
    """Single-winner methods. Used when mode is "single_winner"."""
    primary_method: MultiWinMethod | None = None
    """Narrows `primary_candidates` down to `candidates`.
    Rust default: honest RRV, ranks=25, k=0.5."""
    committee_size: NonNegativeInt | None = None
    """Committee size. Required when mode is "multi_winner"."""
    committee_methods: list[MultiWinMethod] = Field(default_factory=list)
    """Multi-winner methods. Used when mode is "multi_winner"."""

    def to_json(self, **kwargs: Any) -> str:
        """This config as the JSON the Rust side reads: unset fields are left out,
        so Rust fills in its defaults."""
        return self.model_dump_json(exclude_none=True, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Like `to_json`, as a JSON-compatible dict."""
        return self.model_dump(mode="json", exclude_none=True)

    def normalized(self) -> dict[str, Any]:
        """This config as the Rust side sees it, every default filled in.

        Also runs the Rust checks, raising `ValueError` if they fail.
        """
        from ._mcelect import normalize_config

        return json.loads(normalize_config(self.to_json()))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Config:
        """Load a config file: `.toml` files as TOML, everything else as JSON."""
        path = os.fspath(path)
        with open(path, "rb") as f:
            data = tomllib.load(f) if path.endswith(".toml") else json.load(f)
        return cls.model_validate(data)
