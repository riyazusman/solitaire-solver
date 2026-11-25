"""Deterministic baseline benchmark script for TriPeaks, Pyramid, and Freecell.

Run via:
    .venv/Scripts/python.exe tests/benchmarks.py
"""

import random
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solvers.models import Card, STANDARD_DECK, parse_board_and_stock
from solvers.tripeaks import TriPeaksBoard, TriPeaksSolver
from solvers.pyramid import PyramidBoard, PyramidSolver
from solvers.freecell import FreecellState, FreecellSolver
from tests.test_models import ORDERED_DECK_STR


def _deal(seed: int):
    rng = random.Random(seed)
    cards = sorted(STANDARD_DECK)
    rng.shuffle(cards)
    parsed = [Card(c) for c in cards]
    return parsed[:28], parsed[28:]


def _measure(label, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    info = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:44s} | {dt:7.4f}s | peak {peak / 1024:8.1f} KB | {info}", flush=True)


def main():
    print("=" * 100)
    print("Solitaire Solver Studio — Baseline Benchmarks")
    print("=" * 100)

    # TriPeaks
    for seed in (4, 5, 8, 11):
        b, s = _deal(seed)
        _measure(
            f"TriPeaks seed={seed} (fast)",
            lambda b=b, s=s: (
                lambda solver: (
                    lambda res: f"cleared={res[1]}, draws={solver.min_stock_used}, states={solver.states_visited}"
                )(solver.solve())
            )(TriPeaksSolver(b, s, TriPeaksBoard(), wrap_enabled=True, minimize_draws=False)),
        )

    b_ord, s_ord = parse_board_and_stock(ORDERED_DECK_STR)
    b_mod = list(b_ord)
    b_mod[18], b_mod[0] = b_mod[0], b_mod[18]
    b_mod[19], b_mod[1] = b_mod[1], b_mod[19]
    _measure(
        "TriPeaks modified-ordered (minimize_draws)",
        lambda: (
            lambda solver: (
                lambda res: f"cleared={res[1]}, draws={solver.min_stock_used}, states={solver.states_visited}"
            )(solver.solve())
        )(TriPeaksSolver(b_mod, s_ord, TriPeaksBoard(), wrap_enabled=True, minimize_draws=True)),
    )

    # Pyramid
    for seed in (3, 5):
        b, s = _deal(seed)
        _measure(
            f"Pyramid seed={seed} (1-pass fast)",
            lambda b=b, s=s: (
                lambda solver: (
                    lambda res: f"cleared={res[1]}, draws={solver.min_stock_used}, states={solver.states_visited}"
                )(solver.solve())
            )(PyramidSolver(b, s, PyramidBoard(), minimize_draws=False, max_passes=1)),
        )

    # Freecell
    deck_ordered = b_ord + s_ord
    _measure(
        "Freecell ordered (whole board, fast)",
        lambda: (
            lambda solver: (
                lambda res: f"solved={res[1]}, moves={len(res[0])}, states={solver.states_visited}"
            )(solver.solve())
        )(FreecellSolver(FreecellState(deck_ordered))),
    )

    for seed in (1, 2, 3):
        b, s = _deal(seed)
        deck = b + s
        _measure(
            f"Freecell seed={seed} (whole board, fast)",
            lambda d=deck: (
                lambda solver: (
                    lambda res: f"solved={res[1]}, moves={len(res[0])}, states={solver.states_visited}"
                )(solver.solve())
            )(FreecellSolver(FreecellState(d))),
        )

    b1, s1 = _deal(1)
    _measure(
        "Freecell seed=1 (target 2S, min_moves)",
        lambda: (
            lambda solver: (
                lambda res: f"solved={res[1]}, moves={len(res[0])}, states={solver.states_visited}"
            )(solver.solve())
        )(FreecellSolver(FreecellState(b1 + s1), target_card=Card("2S"), minimize_moves=True)),
    )


if __name__ == "__main__":
    main()
