"""Evaluation harness for measuring TOC generation quality against golden data."""

from pdfalive.evaluation.metrics import EvalReport, GoldenEntry, evaluate_toc


__all__ = ["EvalReport", "GoldenEntry", "evaluate_toc"]
