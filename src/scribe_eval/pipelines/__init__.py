"""Choose a pipeline from an experiment's `[pipeline]` table."""

from importlib import import_module
from pathlib import Path

from .base import Pipeline, PipelineResult, Timeline

__all__ = ["Pipeline", "PipelineResult", "Timeline", "create_pipeline"]

_BUILT_IN = {"scribnic": "scribe_eval.pipelines.scribnic:ScribnicPipeline"}


def create_pipeline(
    pipeline_type: str, settings: dict[str, object], *, base_directory: Path
) -> Pipeline:
    """Build a pipeline without loading any model.

    Args:
        pipeline_type: A built-in name, or `package.module:ClassName` for any class
            implementing `Pipeline`.
        settings: The rest of the `[pipeline]` table, passed as keyword arguments.
        base_directory: The experiment's directory, for relative paths.

    Returns:
        The pipeline.

    Raises:
        ValueError: If the type is unknown or its dependencies aren't installed.
    """
    location = _BUILT_IN.get(pipeline_type, pipeline_type)
    if ":" not in location:
        known = ", ".join(sorted(_BUILT_IN))
        raise ValueError(
            f"Unknown pipeline {pipeline_type!r}. "
            f"Use one of: {known}, or package.module:ClassName."
        )
    module_name, class_name = location.split(":", 1)
    try:
        pipeline_class = getattr(import_module(module_name), class_name)
    except ImportError as error:
        raise ValueError(
            f"Can't load pipeline {pipeline_type!r}: {error}. Is its package installed?"
        ) from error
    return pipeline_class(base_directory=base_directory, **settings)
