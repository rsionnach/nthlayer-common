"""Shape checks shared by the manifest parsers (opensrm-wo62.4.28).

``load_manifest`` raises only ``OSError``, ``ManifestLoadError`` or
``ValueError``. A parser that reads a position with ``.get`` or iterates it
used to end in a bare ``AttributeError`` or ``TypeError`` when the file put
some other type there. Each helper returns the value when it has the shape
the caller is about to rely on, and otherwise raises the caller's own parse
error, naming the position.

They refuse exactly what used to crash, and nothing that used to load: a
string where a list of mappings goes iterated to characters, which the
parsers skip, so that file loaded with no entries and still does.
Tightening those is a separate decision.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def shape(value: Any) -> str:
    """What *value* is, for an error message about a wrong shape."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, set | frozenset):
        return "a set"
    if isinstance(value, bytes):
        return "binary data"
    return f"a {type(value).__name__}"


def require_mapping(value: Any, what: str, error: type[Exception]) -> dict[str, Any]:
    """*value* if it is a mapping, else *error* naming *what*."""
    if not isinstance(value, dict):
        raise error(f"{what} must be a mapping, not {shape(value)}")
    return value


def require_iterable(value: Any, what: str, error: type[Exception]) -> Iterable[Any]:
    """*value* if a loop can walk it, else *error* naming *what*.

    For a position the parser iterates, skipping entries that are not
    mappings: a list is what belongs there, but a string, mapping, set or
    binary also iterates (to nothing the parser keeps) and has always
    loaded. A number, boolean, null or date cannot be iterated at all.
    """
    if not isinstance(value, list | str | bytes | dict | set | frozenset):
        raise error(f"{what} must be a list, not {shape(value)}")
    return value


def require_entries(value: Any, what: str, error: type[Exception]) -> list[dict[str, Any]]:
    """*value* as a list of mappings, else *error* naming *what*.

    For a position whose every entry the parser reads as a mapping. As in
    ``require_iterable``, an empty string, mapping, set or binary has no
    entries and has always loaded as none.
    """
    entries = list(require_iterable(value, what, error))
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise error(f"{what}[{i}] must be a mapping, not {shape(entry)}")
    return entries


def require_float(value: Any, what: str, error: type[Exception]) -> float:
    """``float(value)``, else *error* naming *what*.

    For a position the parser has always passed to ``float()``. What that
    accepts is unchanged: ``true`` is still 1.0, and a string that spells
    no number is still the ``ValueError`` ``float()`` raises. A value
    ``float()`` cannot take at all (a list, a mapping, null), or an integer
    too large for one, used to be a bare TypeError or OverflowError.
    """
    try:
        return float(value)
    except TypeError as exc:
        raise error(f"{what} must be a number, not {shape(value)}") from exc
    except OverflowError as exc:
        # An integer too large for a float (YAML integers are unbounded).
        raise error(f"{what} is too large to be a number") from exc


def require_string(value: Any, what: str, error: type[Exception]) -> str:
    """*value* if it is a string, else *error* naming *what*."""
    if not isinstance(value, str):
        raise error(f"{what} must be a string, not {shape(value)}")
    return value
