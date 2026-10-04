"""The pydantic models in `mcelect.config` match the Rust config definitions.

Rust's side comes from its JSON Schema (schemars, built from the same serde
derives that read configs). Python's side comes from the models themselves:
their own JSON Schema can't show the enum tags, which `_Variant` adds in a
validator and serializer. Both are reduced to the same "shape":

* a primitive: ``"number"``, ``"integer"``, ``"uint"`` (non-negative), ``"string"``, ``"boolean"``
* ``{"literal": [values]}`` -- a fixed set of strings
* ``{"array": shape}``
* ``{"object": {field: {"required": bool, "type": shape}}}``
* ``{"enum": {tag: shape}}`` -- an externally tagged enum; a unit variant's shape is None

A field that may be omitted has its nullability dropped: Python spells "omit
it, Rust defaults it" as ``X | None = None``, Rust as a plain ``X`` with a
default, and both mean the same thing here.

Not covered: aliases (schemars ignores them; see test_config.py), default
values, and numeric bounds beyond non-negativity.
"""

from __future__ import annotations

import json
import types
from typing import Annotated, Any, Iterator, Literal, Union, get_args, get_origin

import annotated_types
import pytest
from pydantic import BaseModel, RootModel

from mcelect import _mcelect
from mcelect.config import Config

# --- Python side --------------------------------------------------------------

_PRIMITIVES = {float: "number", int: "integer", str: "string", bool: "boolean"}


def _is_non_negative(metadata: Any) -> bool:
    return any(isinstance(m, annotated_types.Ge) and m.ge == 0 for m in metadata)


def _variant_payload(cls: type) -> Any:
    """The shape inside a `_Variant`/`_RootVariant`'s tag."""
    if issubclass(cls, RootModel):
        return _py_field_type(cls.model_fields["root"])
    return _py_model(cls)


def _py_model(cls: type[BaseModel]) -> Any:
    return {
        "object": {
            name: {"required": field.is_required(), "type": _py_field_type(field)}
            for name, field in cls.model_fields.items()
        }
    }


def _py_field_type(field: Any) -> Any:
    # pydantic moves a field's top-level Annotated metadata (NonNegativeInt's
    # Ge(0)) out of `annotation` and into `metadata`.
    shape = _py_shape(field.annotation, optional=not field.is_required())
    return "uint" if shape == "integer" and _is_non_negative(field.metadata) else shape


def _py_shape(tp: Any, optional: bool = False) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if origin is Annotated:
        inner, metadata = args[0], args[1:]
        shape = _py_shape(inner, optional)
        return "uint" if shape == "integer" and _is_non_negative(metadata) else shape
    if origin in (Union, types.UnionType):
        members = [a for a in args if a is not type(None)]
        if len(members) < len(args) and not optional:
            return {"nullable": _py_union(members)}
        return _py_union(members)
    if origin is Literal:
        return {"literal": sorted(args)}
    if origin is list:
        return {"array": _py_shape(args[0])}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return _py_model(tp)
    return _PRIMITIVES[tp]


def _py_union(members: list[Any]) -> Any:
    if len(members) == 1:
        return _py_shape(members[0])
    # An externally tagged enum: tagged variants (from `_tagged_union`, wrapped
    # in Annotated[v, Tag]) and unit variants, spelled as Literals.
    variants: dict[str, Any] = {}
    for member in members:
        if get_origin(member) is Annotated:
            member = get_args(member)[0]
        if get_origin(member) is Literal:
            variants.update({value: None for value in get_args(member)})
        else:
            variants[member.TAG] = _variant_payload(member)
    return {"enum": variants}


# --- Rust side ----------------------------------------------------------------


def _rust_shape(
    node: dict[str, Any], defs: dict[str, Any], optional: bool = False
) -> Any:
    if "$ref" in node:
        return _rust_shape(defs[node["$ref"].rsplit("/", 1)[1]], defs, optional)

    if "oneOf" in node:
        alternatives = node["oneOf"]
        if all("const" in a for a in alternatives):
            return {"literal": sorted(a["const"] for a in alternatives)}
        variants: dict[str, Any] = {}
        for alt in alternatives:
            if "const" in alt:
                variants[alt["const"]] = None
            else:
                ((tag, payload),) = alt["properties"].items()
                variants[tag] = _rust_shape(payload, defs)
        return {"enum": variants}

    if "enum" in node:
        return {"literal": sorted(node["enum"])}

    kind = node["type"]
    if isinstance(kind, list):
        (kind,) = [k for k in kind if k != "null"]
        if not optional:
            return {"nullable": _rust_shape({**node, "type": kind}, defs)}

    if kind == "object":
        required = set(node.get("required", []))
        return {
            "object": {
                name: {
                    "required": name in required,
                    "type": _rust_shape(prop, defs, optional=name not in required),
                }
                for name, prop in node["properties"].items()
            }
        }
    if kind == "array":
        return {"array": _rust_shape(node["items"], defs)}
    if kind == "integer" and node.get("minimum") == 0:
        return "uint"
    return kind


# --- Comparison ---------------------------------------------------------------


def _diff(rust: Any, py: Any, path: str = "Config") -> Iterator[str]:
    """Each place the shapes disagree, as a readable line."""
    if isinstance(rust, dict) and isinstance(py, dict) and rust.keys() == py.keys():
        if len(rust) == 1 and next(iter(rust)) in ("object", "enum"):
            ((kind, rust_members),) = rust.items()
            py_members = py[kind]
            noun = "field" if kind == "object" else "variant"
            for name in rust_members.keys() - py_members.keys():
                yield f"{path}: {noun} {name!r} is in Rust but not Python"
            for name in py_members.keys() - rust_members.keys():
                yield f"{path}: {noun} {name!r} is in Python but not Rust"
            for name in rust_members.keys() & py_members.keys():
                yield from _diff(rust_members[name], py_members[name], f"{path}.{name}")
            return
        for key in rust:
            yield from _diff(
                rust[key],
                py[key],
                path if key in ("type", "array", "nullable") else f"{path}.{key}",
            )
        return
    if rust != py:
        yield f"{path}: Rust has {rust!r}, Python has {py!r}"


def test_python_models_match_the_rust_config_schema():
    schema = json.loads(_mcelect.config_schema())
    rust = _rust_shape(schema, schema.get("$defs", {}))
    py = _py_model(Config)
    problems = list(_diff(rust, py))
    if problems:
        pytest.fail(
            "mcelect.config has drifted from Rust:\n" + "\n".join(problems),
            pytrace=False,
        )


def test_the_check_notices_drift():
    """A model that has drifted from Rust is caught, with a useful message."""
    schema = json.loads(_mcelect.config_schema())
    rust = _rust_shape(schema, schema.get("$defs", {}))

    class Drifted(Config):
        extra_field: int = 0

    problems = list(_diff(rust, _py_model(Drifted)))
    assert problems == ["Config: field 'extra_field' is in Python but not Rust"]
