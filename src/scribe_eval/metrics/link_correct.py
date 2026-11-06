"""LINK and CORRECT, replicating Wang et al.'s ADS_evaluation.

A judge extracts facts from the generated and reference notes, then decides, fact by
fact, whether a reference fact supports each generated one. With `supported` facts:

    CORRECT = supported / generated facts
    LINK = min(1, supported / reference facts)

This follows their code, not a reinterpretation. ADS_evaluation has no license, so its
prompts aren't copied here: they're fetched verbatim from a pinned commit at first use.
The judge model is swappable, and they used GPT-4o at temperature 0.7.
"""

import ast
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from ..judge import Asker
from ..types import Prediction, Requires, Sample, Score
from .text import mean_values

_COMMIT = "392de464b50d12445b2385a7735a2d3f8f26e8cf"
_SOURCE = f"https://raw.githubusercontent.com/AutoEvaluation/ADS_evaluation/{_COMMIT}"
_CACHE = Path.home() / ".cache" / "scribe_eval" / "ads_evaluation" / _COMMIT
_FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_id": {"type": "integer"},
                    "content": {"type": "string"},
                },
                "required": ["fact_id", "content"],
            },
        }
    },
    "required": ["facts"],
}
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "string", "enum": ["YES", "NO"]},
        "supporting_claim": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["supported", "supporting_claim", "explanation"],
}


def prompt_from_source(source: str, name: str) -> str:
    """Find a string assigned to a name or attribute in Python source.

    Raises:
        ValueError: If the source doesn't assign that name.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            target_name = getattr(target, "id", getattr(target, "attr", None))
            if target_name == name:
                return ast.literal_eval(node.value)
    raise ValueError(f"No {name} in ADS_evaluation's source.")


def _fetch(relative: str) -> str:
    """Read one of their files, downloading it once."""
    path = _CACHE / Path(relative).name
    if not path.exists():
        request = urllib.request.Request(
            f"{_SOURCE}/{relative}", headers={"User-Agent": "scribe-eval"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _prompts() -> tuple[str, str]:
    """Load their extraction and evaluation prompts."""
    extractor = _fetch("auto_metrics/fact/fact_extractor.py")
    evaluator = _fetch("auto_metrics/fact/llm_fact_eval.py")
    return (
        prompt_from_source(extractor, "FACT_PROMPT"),
        prompt_from_source(evaluator, "EVAL_PROMPT"),
    )


def _extract(judge: Asker, text: str) -> list[str]:
    """List a note's atomic facts."""
    prompt = _prompts()[0].format(input=text)
    return [fact["content"] for fact in judge.ask(prompt, _FACTS_SCHEMA)["facts"]]


def _count_supported(
    judge: Asker, generated: Sequence[str], reference: Sequence[str]
) -> int:
    """Ask about each generated fact, one call each, as they do."""
    listed = "\n".join(f"- {fact}" for fact in reference)
    supported = 0
    for fact in generated:
        prompt = _prompts()[1].format(reference_facts=listed, pred_fact=fact)
        supported += judge.ask(prompt, _VERDICT_SCHEMA)["supported"] == "YES"
    return supported


def _counts(judge: Asker, prediction: Prediction, sample: Sample) -> dict[str, float]:
    """Compare both notes. The judge's cache makes the second metric free."""
    generated = _extract(judge, prediction.artifacts["note"].text)
    reference = _extract(judge, sample.artifact("note").text)
    return {
        "supported": float(_count_supported(judge, generated, reference)),
        "generated": float(len(generated)),
        "reference": float(len(reference)),
    }


@dataclass(frozen=True)
class LinkCorrect:
    """One of the two scores. Needs a judge to run."""

    name: str
    group: str = "link_correct"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))
    judge: Asker | None = None

    def with_judge(self, judge: Asker) -> "LinkCorrect":
        """Return a copy that can score."""
        return replace(self, judge=judge)

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Compute the score, which needs facts on both sides."""
        counts = _counts(self.judge, prediction, sample)
        if not counts["generated"] or not counts["reference"]:
            return None
        if self.name == "link":
            value = min(1.0, counts["supported"] / counts["reference"])
        else:
            value = counts["supported"] / counts["generated"]
        return Score(self.name, value, counts)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average scores, as the paper reports them."""
        return mean_values(scores)
