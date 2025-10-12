"""Automated CLI walkthrough demo for PERSIST.

This script simulates a user interacting with ``main.py`` so we can
capture the Rich-rendered panels without manual input. It feeds a set of
curated answers into the Questionary prompts and prints the resulting
output to stdout. The script is useful for documentation and for
producing screenshots of the guided CLI experience.
"""

from __future__ import annotations

from contextlib import contextmanager
import pathlib
import sys
from types import SimpleNamespace
from typing import Iterable, Iterator, List

import questionary

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as persist_main


class ResponseQueue:
    """Simple iterator wrapper that raises a helpful error when exhausted."""

    def __init__(self, values: Iterable):
        self._values: Iterator = iter(values)
        self._history: List = []

    def next(self, message: str, kind: str):
        try:
            value = next(self._values)
        except StopIteration as exc:
            raise RuntimeError(
                f"Ran out of predefined {kind} responses while prompting for: '{message}'"
            ) from exc

        self._history.append((kind, message, value))
        return value


@contextmanager
def patched_questionary(*, select_answers, confirm_answers, text_answers):
    """Temporarily patch Questionary helpers to return canned answers."""

    select_queue = ResponseQueue(select_answers)
    confirm_queue = ResponseQueue(confirm_answers)
    text_queue = ResponseQueue(text_answers)

    def _stub_factory(queue: ResponseQueue, kind: str):
        def _stub(message: str, **kwargs):  # type: ignore[override]
            value = queue.next(message, kind)
            print(f"[auto-response] {message} -> {value}")
            return SimpleNamespace(ask=lambda: value)

        return _stub

    original_select = questionary.select
    original_confirm = questionary.confirm
    original_text = questionary.text

    questionary.select = _stub_factory(select_queue, "select")  # type: ignore[assignment]
    questionary.confirm = _stub_factory(confirm_queue, "confirm")  # type: ignore[assignment]
    questionary.text = _stub_factory(text_queue, "text")  # type: ignore[assignment]

    try:
        yield
    finally:
        questionary.select = original_select  # type: ignore[assignment]
        questionary.confirm = original_confirm  # type: ignore[assignment]
        questionary.text = original_text  # type: ignore[assignment]


def run_demo() -> None:
    """Execute ``main.main`` with a deterministic set of responses."""

    select_answers = [
        "Run a standard single-agent experiment (Recommended for beginners)",
    ]

    confirm_answers = [
        True,   # Review core training hyperparameters
        True,   # Adjust model capacity
        False,  # Skip curriculum tweaks
        False,  # Do not start a full training run
    ]

    text_answers = [
        "150",      # Number of training epochs
        "250000",   # Max environment steps
        "32",       # Gradient updates cadence
        "12",       # Model-based rollouts per update
        "12000",    # Switch shield to amortized mode
        "256",      # Batch size
        "0.985",    # Discount factor (gamma)
        "0.01",     # Target network decay (tau)
        "0.00025",  # Actor learning rate
        "0.00035",  # Critic learning rate
        "4000",     # Checkpoint cadence
        "196",      # State estimator hidden dimension
        "3",        # State estimator layers
        "320",      # Empowerment hidden dimension
        "5",        # Empowerment rollout depth
        "160",      # Safety network hidden dimension
        "0.0002",   # Safety network learning rate
    ]

    with patched_questionary(
        select_answers=select_answers,
        confirm_answers=confirm_answers,
        text_answers=text_answers,
    ):
        persist_main.main()


if __name__ == "__main__":
    run_demo()
