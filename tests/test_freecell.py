import random
import pytest

from solvers.models import Card, STANDARD_DECK, parse_deck
from solvers.freecell import FreecellState, FreecellSolver
from tests.test_models import ORDERED_DECK_STR
from tests.verifiers import verify_freecell_solution


def _ordered_deck():
    return parse_deck(ORDERED_DECK_STR)


def _shuffled_deck(seed: int):
    rng = random.Random(seed)
    cards = sorted(STANDARD_DECK)
    rng.shuffle(cards)
    return [Card(c) for c in cards]


def _auto_clear_deck():
    """Deck dealt K..A down every cascade so all 52 cards auto-foundation immediately."""
    cards = []
    for rank in ("K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2", "A"):
        for suit in ("S", "H", "D", "C"):
            cards.append(Card(rank + suit))
    return cards


def test_freecell_whole_board_ordered_deck_verified():
    deck = _ordered_deck()
    solver = FreecellSolver(FreecellState(deck))
    seq, solved, _ = solver.solve()
    assert solved
    res = verify_freecell_solution(deck, seq, expect_solved=True)
    assert res["goal_met"]
    assert res["moves"] == len(seq)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_freecell_whole_board_seeded_deals_verified(seed):
    deck = _shuffled_deck(seed)
    solver = FreecellSolver(FreecellState(deck))
    seq, solved, _ = solver.solve()
    assert solved
    res = verify_freecell_solution(deck, seq, expect_solved=True)
    assert res["goal_met"]


def test_freecell_target_card_goal_verified():
    deck = _ordered_deck()
    target = Card("2S")
    solver = FreecellSolver(FreecellState(deck), target_card=target, minimize_moves=False)
    seq, solved, _ = solver.solve()
    assert solved
    res = verify_freecell_solution(deck, seq, target_card=target, expect_solved=True)
    assert res["goal_met"]


def test_freecell_target_ranks_goal_verified():
    deck = _ordered_deck()
    solver = FreecellSolver(FreecellState(deck), target_ranks=[1, 2], rank_count=2)
    seq, solved, _ = solver.solve()
    assert solved
    res = verify_freecell_solution(
        deck, seq, target_ranks=[1, 2], rank_count=2, expect_solved=True
    )
    assert res["goal_met"]


def test_freecell_target_card_minimize_moves_proves_optimal():
    deck = _auto_clear_deck()
    target = Card("3S")
    solver = FreecellSolver(FreecellState(deck), target_card=target, minimize_moves=True)
    seq, solved, proven = solver.solve()
    assert solved and proven
    assert len(seq) == 3  # AS, 2S, 3S straight to foundation
    verify_freecell_solution(deck, seq, target_card=target, expect_solved=True)


def test_freecell_unsolved_move_limit_returns_nonempty_partial_sequence():
    # Put AS at the bottom of Cascade 1 (exposed), while 2S/3S are buried near the top.
    # With target_card=3S and move_limit=4, 3S is unreachable within 4 moves,
    # but AS can be moved to the foundation (remaining drops from 3 to 2).
    deck = _ordered_deck()
    # In ordered deck, AS is at idx 0 (top of Cascade 1) and KC is at idx 51 (bottom of Cascade 4).
    # Swap AS (idx 0) with idx 48 (bottom of Cascade 1: 49th card, 0-based 48).
    deck[0], deck[48] = deck[48], deck[0]
    solver = FreecellSolver(FreecellState(deck), target_card=Card("3S"), move_limit=4)
    seq, solved, proven = solver.solve()
    assert not solved
    assert proven
    assert len(seq) > 0, "Expected best partial sequence when target cannot be reached within move limit"
    res = verify_freecell_solution(deck, seq, target_card=Card("3S"), expect_solved=False)
    assert res["foundations"]["S"] >= 1


def test_freecell_initial_auto_foundation_respects_move_limit():
    deck = _auto_clear_deck()
    solver = FreecellSolver(FreecellState(deck), move_limit=5)
    seq, solved, proven = solver.solve()
    assert not solved, f"Auto-clear returned {len(seq)} moves despite move_limit=5"
    assert proven
    assert len(seq) <= 5
    verify_freecell_solution(deck, seq, expect_solved=False)


def test_freecell_goal_remaining_ignores_already_satisfied_ranks():
    state = FreecellState(_ordered_deck())
    state.foundations = {"S": 2, "H": 2, "D": 0, "C": 0}
    solver = FreecellSolver(state, target_ranks=[2, 10], rank_count=2)
    # Two 2s are already satisfied; reaching two 10s requires at least (10-2)+(10-2) = 16 moves.
    assert solver._goal_remaining() == 16


def test_freecell_state_limit_cap_returns_valid_partial():
    deck = _ordered_deck()
    solver = FreecellSolver(FreecellState(deck), max_states=50)
    seq, solved, proven = solver.solve()
    assert not solved
    assert not proven
    assert solver.state_limit_reached is True
    verify_freecell_solution(deck, seq, expect_solved=False)
