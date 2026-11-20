"""Check the Berta pipeline with a fake Berta and a fake llama-server.

The fakes run as real processes, so these tests cover process ownership, the
worker's protocol, and telemetry without Berta, models, or a network. The checks
against a real Berta checkout skip when `vendor/berta/web-api` is missing, which
`scripts/setup_berta.sh` creates.
"""

import ast
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from scribe_eval import run_experiment
from scribe_eval.artifacts import AudioFile, Transcript
from scribe_eval.pipelines import create_pipeline
from scribe_eval.pipelines.berta import BertaPipeline, worker
from scribe_eval.pipelines.berta.pipeline import _request
from scribe_eval.pipelines.berta.processes import parse_metrics
from scribe_eval.pipelines.berta.telemetry import note_timings
from scribe_eval.storage import read_jsonl

from .helpers import write_tiny_dataset

REAL_BERTA = Path(__file__).resolve().parents[1] / "vendor" / "berta" / "web-api"

# Just enough of Berta's `app` package for the worker: the same module names and
# call signatures, with notes written by the fake llama-server.
FAKE_APP = {
    "app/__init__.py": "",
    "app/config/__init__.py": """
class Settings:
    APP_VERSION = "fake"
    BUILTIN_NOTETYPES_FOLDER = ".prompts/builtin-note-types"
    DEFAULT_AUDIO_BITRATE = "96k"


settings = Settings()
""",
    "app/services/__init__.py": "",
    "app/services/whisperx.py": """
class WhisperXConfig:
    def __init__(self, **values):
        self.__dict__.update(values)


class WhisperXTranscriptionService:
    service_name = "WhisperX"

    def __init__(self):
        self.config = WhisperXConfig(
            device="cpu", model_version="large-v3", compute_type="float32"
        )
""",
    "app/services/audio_processing.py": """
import io


def reformat_audio(original, format, bitrate):
    return io.BytesIO(original.read()), 0
""",
    "app/config/ai.py": """
import json
import os
import urllib.request

from app.services.whisperx import WhisperXTranscriptionService


class Model:
    def __init__(self, name):
        self.name = name


class LlamaCpp:
    def __init__(self, url):
        with urllib.request.urlopen(url + "/v1/models") as response:
            served = json.load(response)["data"]
        self.models = [Model(entry["id"]) for entry in served]


transcription_service = WhisperXTranscriptionService()
generative_ai_services = [LlamaCpp(os.environ["LLAMA_CPP_SERVER_URL"])]
""",
    "app/tasks/__init__.py": """
import json
import os
import urllib.request
from pathlib import Path


class Output:
    def __init__(self, **values):
        self.__dict__.update(values)


async def transcribe_audio(audio, filename, content_type):
    assert content_type == "audio/mpeg" and audio.read()
    return Output(transcript=f"hello {Path(filename).stem}")


def generate_note(model, instructions, context, transcript, output_type):
    if transcript == "crash":
        os._exit(3)
    if transcript.endswith("b"):
        raise RuntimeError("Note rejected")
    request = urllib.request.Request(
        os.environ["LLAMA_CPP_SERVER_URL"] + "/v1/chat/completions",
        json.dumps({"model": model, "content": instructions}).encode(),
    )
    with urllib.request.urlopen(request) as response:
        text = json.load(response)["text"]
    return Output(text=text, promptTokens=100, completionTokens=20)
""",
    ".prompts/builtin-note-types/Common/Full Visit.txt": "Write a note.\n",
}

# Serves /health, /v1/models, /metrics, and a note, counting work like llama-server.
FAKE_LLAMA_SERVER = """
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

if "--version" in sys.argv:
    print("version: fake (build 1)")
    sys.exit(0)
argv = sys.argv
port = int(argv[argv.index("--port") + 1])
alias = argv[argv.index("--alias") + 1]
counters = {
    "prompt_tokens_total": 0,
    "prompt_seconds_total": 0.0,
    "tokens_predicted_total": 0,
    "tokens_predicted_seconds_total": 0.0,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, content_type="application/json"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply(json.dumps({"status": "ok"}))
        elif self.path == "/v1/models":
            self.reply(json.dumps({"data": [{"id": alias}]}))
        else:
            lines = [f"llamacpp:{name} {value}" for name, value in counters.items()]
            self.reply("\\n".join(["# HELP fake", *lines]), "text/plain")

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        counters["prompt_tokens_total"] += 80
        counters["prompt_seconds_total"] += 2.0
        counters["tokens_predicted_total"] += 20
        counters["tokens_predicted_seconds_total"] += 1.0
        self.reply(json.dumps({"text": "S — a\\nO — b\\nA — c\\nP — d"}))


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

EXPERIMENT = """
name = "tiny-berta"

[pipeline]
type = "berta"
berta_directory = "web-api"
llama_server = "bin/llama-server"
note_model = "models/Fake-Note-Q8_0.gguf"
python = "{python}"

[[datasets]]
name = "tiny"
split = "all"
input = "audio"
metrics = ["transcription", "note_format", "resources"]
"""


def write_fake_berta(directory: Path) -> None:
    """Write the fake `app` package, llama-server, and model file."""
    for name, source in FAKE_APP.items():
        path = directory / "web-api" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    script = directory / "bin" / "fake_llama_server.py"
    script.parent.mkdir(parents=True)
    script.write_text(FAKE_LLAMA_SERVER, encoding="utf-8")
    launcher = directory / "bin" / "llama-server"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    model = directory / "models" / "Fake-Note-Q8_0.gguf"
    model.parent.mkdir()
    model.write_bytes(b"GGUF")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        write_fake_berta(self.directory)
        write_tiny_dataset(self.directory / "data" / "tiny", ["a", "b"])
        self.experiment = self.directory / "experiment.toml"
        self.experiment.write_text(
            EXPERIMENT.format(python=sys.executable), encoding="utf-8"
        )
        self.run_directory = self.directory / "run"

    def settings(self, **changes: object) -> dict[str, object]:
        """Point every path at the fakes, as an experiment's table would."""
        return {
            "berta_directory": "web-api",
            "llama_server": "bin/llama-server",
            "note_model": "models/Fake-Note-Q8_0.gguf",
            "python": sys.executable,
            **changes,
        }

    def pipeline(self, **changes: object) -> BertaPipeline:
        """Build the real pipeline on the fakes."""
        return BertaPipeline(base_directory=self.directory, **self.settings(**changes))

    def run_it(self) -> dict:
        """Run the experiment through the fake Berta."""
        _, results = run_experiment(
            self.experiment,
            data_directory=self.directory / "data",
            run_directory=self.run_directory,
        )
        return results

    def rows(self) -> list[dict]:
        """Read predictions."""
        path = self.run_directory / "datasets" / "tiny" / "predictions.jsonl"
        return read_jsonl(path)

    def test_is_a_built_in_pipeline(self):
        pipeline = create_pipeline(
            "berta", self.settings(), base_directory=self.directory
        )
        self.assertIsInstance(pipeline, BertaPipeline)

    def test_missing_paths_fail_before_anything_starts(self):
        with self.assertRaises(ValueError):
            self.pipeline(note_model="models/missing.gguf")

    def test_produces_excludes_the_input(self):
        pipeline = self.pipeline()
        self.assertEqual(pipeline.produces("audio"), {"transcript", "note"})
        self.assertEqual(pipeline.produces("diarized_transcript"), {"note"})
        with self.assertRaises(ValueError):
            pipeline.produces("note")

    def test_describe_reports_both_models_and_no_port(self):
        description = self.pipeline(whisper_model="tiny").describe()
        self.assertEqual(description["transcription"]["model"], "tiny")
        self.assertEqual(description["note"]["model"], "Fake-Note-Q8_0")
        self.assertNotIn("--port", description["note"]["llama_server"])

    def test_runs_berta_tasks_with_the_requested_whisper_model(self):
        self.run_it()
        startup = read_jsonl(self.run_directory / "startup.jsonl")[0]
        self.assertEqual(startup["whisper_model"], "small")
        self.assertEqual(startup["llama_server_version"], "fake (build 1)")
        row = self.rows()[0]
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["artifacts"], ["note", "transcript"])

    def test_failure_keeps_partial_artifacts(self):
        results = self.run_it()
        self.assertEqual(results["datasets"]["tiny"]["n_failed"], 1)
        failed = self.rows()[1]
        self.assertEqual(failed["error"], "RuntimeError: Note rejected")
        self.assertEqual(failed["artifacts"], ["transcript"])
        self.assertEqual(results["datasets"]["tiny"]["metrics"]["wer"]["n"], 2)

    def test_telemetry_uses_shared_phases_and_server_counters(self):
        self.run_it()
        telemetry = self.rows()[0]["telemetry"]
        phases = [(phase["name"], phase["parent"]) for phase in telemetry["phases"]]
        self.assertEqual(
            phases,
            [
                ("transcription", None),
                ("note", None),
                ("prefill", "note"),
                ("generation", "note"),
            ],
        )
        self.assertEqual(telemetry["phases"][2]["seconds"], 2.0)
        model = telemetry["models"][0]
        self.assertEqual(model["prompt_tokens"], 100)
        self.assertEqual(model["prefill_tokens_per_second"], 40.0)
        names = {process["name"] for process in telemetry["processes"]}
        self.assertEqual(names, {"llama-server", "berta-worker"})
        self.assertGreater(telemetry["peak_rss_bytes"], 0)

    def test_closing_stops_every_process(self):
        pipeline = self.pipeline()
        with pipeline.open(self.directory / "telemetry"):
            pipeline.prepare("audio")
            processes = [pipeline._server.process, pipeline._worker.process]
            self.assertTrue(all(process.poll() is None for process in processes))
        self.assertTrue(all(process.poll() is not None for process in processes))

    def test_timeline_is_read_back_for_the_report(self):
        pipeline = self.pipeline()
        telemetry_directory = self.directory / "telemetry"
        audio = self.directory / "data" / "tiny" / "samples" / "a" / "audio.wav"
        with pipeline.open(telemetry_directory):
            pipeline.prepare("audio")
            result = pipeline.run(AudioFile(audio), self.directory / "out")
        timeline = pipeline.timeline([telemetry_directory], result.telemetry)
        self.assertEqual(
            [name for name, _, _ in timeline.windows], ["transcription", "note"]
        )
        self.assertEqual(
            {series.process for series in timeline.memory},
            {"llama-server", "berta-worker"},
        )
        saved = json.loads((self.directory / "out" / "berta.json").read_text())
        self.assertEqual(saved["transcript"], "hello a")

    def test_a_transcript_input_yields_only_a_note(self):
        pipeline = self.pipeline()
        with pipeline.open(self.directory / "telemetry"):
            pipeline.prepare("transcript")
            result = pipeline.run(
                Transcript.from_text("hello a"), self.directory / "out"
            )
        self.assertEqual(set(result.artifacts), {"note"})
        self.assertEqual([phase.name for phase in result.telemetry.phases][0], "note")

    def test_a_dead_worker_fails_its_sample_then_stops_the_run(self):
        pipeline = self.pipeline()
        with pipeline.open(self.directory / "telemetry"):
            pipeline.prepare("transcript")
            result = pipeline.run(Transcript.from_text("crash"), self.directory / "a")
            self.assertEqual(result.status, "error")
            self.assertIn("exited with code 3", result.error)
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                pipeline.run(Transcript.from_text("hello"), self.directory / "b")

    def test_a_load_failure_stops_the_run_and_releases_the_server(self):
        pipeline = self.pipeline(note_type="Missing Note")
        with pipeline.open(self.directory / "telemetry"):
            with self.assertRaisesRegex(RuntimeError, "Missing Note"):
                pipeline.prepare("audio")
            self.assertIsNone(pipeline._worker)
        self.assertIsNone(pipeline._server)


class RequestTests(unittest.TestCase):
    def test_relative_audio_paths_become_absolute(self):
        request = _request(AudioFile(Path("data/sample/audio.wav")))
        self.assertTrue(Path(request["audio"]).is_absolute())
        self.assertEqual(Path(request["audio"]), Path.cwd() / "data/sample/audio.wav")


class TelemetryTests(unittest.TestCase):
    def test_metrics_are_parsed_without_the_prefix(self):
        text = "# HELP x\nllamacpp:prompt_seconds_total 1.5\nbroken\n"
        self.assertEqual(parse_metrics(text), {"prompt_seconds_total": 1.5})

    def test_note_timings_are_the_difference_between_snapshots(self):
        before = {
            "prompt_tokens_total": 10,
            "prompt_seconds_total": 1.0,
            "tokens_predicted_total": 5,
            "tokens_predicted_seconds_total": 1.0,
        }
        after = {
            "prompt_tokens_total": 110,
            "prompt_seconds_total": 3.0,
            "tokens_predicted_total": 25,
            "tokens_predicted_seconds_total": 2.0,
        }
        timings = note_timings(before, after)
        self.assertEqual(timings.prefill_seconds, 2.0)
        self.assertEqual(timings.prefill_tokens_per_second, 50.0)
        self.assertEqual(timings.generation_tokens_per_second, 20.0)
        self.assertIsNone(note_timings({}, after))
        self.assertIsNone(note_timings(after, after))


@unittest.skipUnless(REAL_BERTA.is_dir(), "No Berta checkout in vendor/berta")
class BertaCheckoutTests(unittest.TestCase):
    """Check, without importing Berta, the code the worker and `describe` rely on."""

    def source(self, name: str) -> ast.Module:
        """Parse one of Berta's modules."""
        return ast.parse((REAL_BERTA / name).read_text(encoding="utf-8"))

    def test_plain_text_is_a_berta_output_type(self):
        text = (REAL_BERTA / "app/schemas/note_output_types.py").read_text()
        self.assertIn(f'"{worker.OUTPUT_TYPE}"', text)

    def test_the_default_note_type_exists(self):
        folder = REAL_BERTA / ".prompts" / "builtin-note-types"
        self.assertIn(
            "Impression and Plan", worker._note_instructions(folder, "Full Visit")
        )

    def test_generate_note_takes_arguments_in_the_order_the_worker_passes(self):
        module = self.source("app/tasks/generation.py")
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_note"
        )
        names = [argument.arg for argument in function.args.args]
        self.assertEqual(
            names, ["model", "instructions", "context", "transcript", "output_type"]
        )

    def test_whisperx_passes_the_values_the_worker_replaces(self):
        text = (REAL_BERTA / "app/services/whisperx.py").read_text()
        self.assertIn('model_version="large-v3"', text)
        self.assertIn('compute_type="float32"', text)

    def test_the_llama_cpp_adapter_sends_the_reported_sampling(self):
        text = (REAL_BERTA / "app/services/llama_cpp.py").read_text()
        self.assertIn("temperature: int = 0", text)
        self.assertIn('"max_tokens": 4096', text)
