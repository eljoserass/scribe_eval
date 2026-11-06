"""Ask a language model for structured answers, and remember them.

Any OpenAI-compatible server works. Answers are cached in the run, so rescoring costs
nothing and a run can be scored again after an interruption.
"""

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from .storage import append_jsonl, read_jsonl

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TIMEOUT_SECONDS = 900


class Asker(Protocol):
    """Answer a prompt with JSON matching a schema."""

    def ask(self, prompt: str, schema: dict) -> dict:
        """Return the parsed answer.

        Raises:
            ValueError: If there is no valid answer. A judge that failed must never
                read as a judgement.
        """
        ...


@dataclass(frozen=True)
class JudgeConfig:
    """Where the judge runs. Clinical text only leaves the machine if allowed."""

    base_url: str
    model: str
    allow_remote: bool = False
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse to send text to another machine by accident."""
        host = urlparse(self.base_url).hostname
        if host not in _LOCAL_HOSTS and not self.allow_remote:
            raise ValueError(
                f"Judge at {host} is not local, and prompts contain clinical text. "
                "Set allow_remote = true if that is acceptable."
            )


class Judge:
    """A model behind an HTTP endpoint, with answers cached on disk."""

    def __init__(self, config: JudgeConfig, cache_path: Path) -> None:
        """Keep the settings. The cache is read on first use."""
        self.config = config
        self.cache_path = cache_path
        self._answers: dict[str, dict] | None = None

    def _cached(self) -> dict[str, dict]:
        """Load earlier answers once."""
        if self._answers is None:
            rows = read_jsonl(self.cache_path)
            self._answers = {row["key"]: row["answer"] for row in rows}
        return self._answers

    def _key(self, prompt: str, schema: dict) -> str:
        """Identify a question, including who is asked."""
        data = [self.config.model, self.config.options, prompt, schema]
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _request(self, prompt: str, schema: dict) -> dict:
        """Call the model, asking the server to enforce the schema."""
        body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
            **self.config.options,
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except (OSError, KeyError, IndexError, json.JSONDecodeError) as error:
            raise ValueError(f"The judge gave no valid answer: {error}") from error

    def ask(self, prompt: str, schema: dict) -> dict:
        """Answer from the cache, or ask the model and remember the result."""
        key = self._key(prompt, schema)
        answers = self._cached()
        if key not in answers:
            answers[key] = self._request(prompt, schema)
            append_jsonl(self.cache_path, {"key": key, "answer": answers[key]})
        return answers[key]
