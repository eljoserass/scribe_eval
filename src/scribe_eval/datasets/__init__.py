"""Registered converters from raw datasets to the common prepared format."""

from collections.abc import Callable

from . import primock57, teme

Converter = Callable[..., None]

CONVERTERS: dict[str, Converter] = {
    "primock57": primock57.prepare,
    "smoke": teme.prepare_smoke,
}
