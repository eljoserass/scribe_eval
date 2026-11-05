"""Turn a dataset entry's selection into concrete sample IDs."""

import math
import random
from collections.abc import Sequence

from .config import DatasetEntry


def select_ids(entry: DatasetEntry, available: Sequence[str]) -> tuple[str, ...]:
    """Choose which samples of a split to evaluate.

    Args:
        entry: Selection settings; `ids`, `limit`, and `fraction` are exclusive.
        available: Every sample ID in the split.

    Returns:
        Sorted IDs. The same seed and split always give the same subset.

    Raises:
        ValueError: If a requested ID is missing, or nothing is selected.
    """
    ordered = sorted(available)
    if entry.ids is not None:
        missing = sorted(set(entry.ids) - set(ordered))
        if missing:
            raise ValueError(f"Unknown sample IDs: {', '.join(missing)}")
        selected = sorted(set(entry.ids))
    elif entry.limit is not None or entry.fraction is not None:
        count = entry.limit or math.ceil(entry.fraction * len(ordered))
        count = min(count, len(ordered))
        selected = sorted(random.Random(entry.seed).sample(ordered, count))
    else:
        selected = ordered
    if not selected:
        raise ValueError(f"No samples selected for {entry.name!r}.")
    return tuple(selected)
