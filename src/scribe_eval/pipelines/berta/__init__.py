"""Evaluate Berta AI Scribe. The only part of scribe_eval that runs Berta.

Berta runs in its own environment, in a worker process; nothing here imports it.
"""

from .pipeline import BertaPipeline

__all__ = ["BertaPipeline"]
