"""Read and write the common on-disk dataset format."""

import re
from dataclasses import dataclass
from pathlib import Path

from scribnic import Artifact, ArtifactKind, AudioFile, load_artifact, save_artifact

from ..storage import read_json, write_json
from ..types import Sample

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_ARTIFACT_FILES: dict[ArtifactKind, str] = {
    "transcript": "transcript.json",
    "diarized_transcript": "diarized_transcript.json",
    "note": "note.json",
}


def sample_directory(root: Path, sample_id: str) -> Path:
    """Locate a sample, rejecting IDs unsafe as directory names."""
    if not _SAFE_ID.fullmatch(sample_id):
        raise ValueError(f"Unsafe sample ID: {sample_id!r}")
    return root / "samples" / sample_id


def write_sample(
    root: Path, sample_id: str, artifacts: list[Artifact], metadata: dict[str, object]
) -> None:
    """Save a sample's JSON artifacts and metadata; audio goes in as `audio.wav`."""
    directory = sample_directory(root, sample_id)
    directory.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        save_artifact(artifact, directory / _ARTIFACT_FILES[artifact.kind])
    write_json(directory / "metadata.json", metadata)


def write_dataset(
    root: Path, *, name: str, version: str, language: str, license: str, source: str
) -> None:
    """Describe a dataset from the samples on disk, in a single `all` split."""
    samples = sorted(path for path in (root / "samples").iterdir() if path.is_dir())
    provides = set()
    for sample in samples:
        if (sample / "audio.wav").exists():
            provides.add("audio")
        provides.update(
            kind
            for kind, filename in _ARTIFACT_FILES.items()
            if (sample / filename).exists()
        )
    write_json(
        root / "dataset.json",
        {
            "name": name,
            "version": version,
            "language": language,
            "license": license,
            "source": source,
            "provides": sorted(provides),
            "splits": {"all": [sample.name for sample in samples]},
        },
    )


@dataclass(frozen=True)
class PreparedDataset:
    """Dataset in the common format. Holds no open resources."""

    root: Path

    def __post_init__(self) -> None:
        """Fail early with a useful message when the data was not prepared."""
        if not (self.root / "dataset.json").exists():
            raise ValueError(
                f"Dataset not prepared: {self.root}. "
                f"Run `scribe-eval prepare {self.root.name}`."
            )

    @property
    def _description(self) -> dict:
        """Read the dataset description."""
        return read_json(self.root / "dataset.json")

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._description["name"]

    @property
    def version(self) -> str:
        """Source version, such as a pinned commit."""
        return self._description["version"]

    @property
    def language(self) -> str:
        """Language code of the consultations."""
        return self._description["language"]

    @property
    def license(self) -> str:
        """License of the source data."""
        return self._description["license"]

    @property
    def provides(self) -> set[ArtifactKind]:
        """Artifact kinds available in this dataset."""
        return set(self._description["provides"])

    @property
    def splits(self) -> dict[str, list[str]]:
        """Sample IDs by split."""
        return self._description["splits"]

    def sample(self, sample_id: str) -> Sample:
        """Load one sample's references.

        Raises:
            KeyError: If the sample is not in any split.
        """
        for split, ids in self.splits.items():
            if sample_id in ids:
                return self._load(sample_id, split)
        raise KeyError(f"Unknown sample {sample_id!r} in {self.name!r}.")

    def _load(self, sample_id: str, split: str) -> Sample:
        """Read a sample's artifact files."""
        directory = sample_directory(self.root, sample_id)
        artifacts: dict[ArtifactKind, Artifact] = {}
        if (directory / "audio.wav").exists():
            artifacts["audio"] = AudioFile(directory / "audio.wav")
        for kind, filename in _ARTIFACT_FILES.items():
            if (directory / filename).exists():
                artifacts[kind] = load_artifact(directory / filename)
        return Sample(
            sample_id,
            split,
            self.language,
            artifacts,
            read_json(directory / "metadata.json"),
        )
