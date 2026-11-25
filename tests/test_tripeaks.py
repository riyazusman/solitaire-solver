import random

from solvers.models import Card, STANDARD_DECK, parse_board_and_stock
from solvers.tripeaks import TriPeaksBoard, TriPeaksSolver
from tests.test_models import ORDERED_DECK_STR
from tests.verifiers import verify_tripeaks_solution


def _shuffled_deck(seed: int):
    rng = random.Random(seed)
    cards = sorted(STANDARD_DECK)
    rng.shuffle(cards)
    parsed = [Card(c) for c in cards]
    return parsed[:28], parsed[28:]


def test_tripeaks_board_geometry():
    board = TriPeaksBoard()
    assert board.get_initial_free_cards() == list(range(18, 28))
    assert board.get_row_and_position(0) == (1, 1)
    assert board.get_row_and_position(2) == (1, 3)
    assert board.get_row_and_position(3) == (2, 1)
    assert board.get_row_and_position(9) == (3, 1)
    assert board.get_row_and_position(18) == (4, 1)
    assert board.get_row_and_position(27) == (4, 10)


def test_tripeaks_ordered_deck_clears_in_zero_draws():
    board_cards, stock_cards = parse_board_and_stock(ORDERED_DECK_STR)
    solver = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=True)
    seq, cleared = solver.solve()
    assert cleared == 28
    assert solver.min_stock_used == 0
    result = verify_tripeaks_solution(board_cards, stock_cards, seq, wrap_enabled=True)
    assert result["cleared"] == 28
    assert result["draws"] == 0


def test_tripeaks_wrap_disabled_still_verified():
    board_cards, stock_cards = _shuffled_deck(42)
    solver = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=False)
    seq, cleared = solver.solve()
    result = verify_tripeaks_solution(board_cards, stock_cards, seq, wrap_enabled=False)
    assert result["cleared"] == cleared
    assert result["draws"] == solver.min_stock_used


def test_tripeaks_minimize_draws_never_worse_than_fast_solve():
    board_cards, stock_cards = parse_board_and_stock(ORDERED_DECK_STR)
    # Swap two bottom cards with top cards so clearing requires stock draws
    board_cards = list(board_cards)
    board_cards[18], board_cards[0] = board_cards[0], board_cards[18]
    board_cards[19], board_cards[1] = board_cards[1], board_cards[19]

    fast = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=True, minimize_draws=False)
    fast_seq, fast_cleared = fast.solve()
    fast_res = verify_tripeaks_solution(board_cards, stock_cards, fast_seq, wrap_enabled=True)
    assert fast_res["cleared"] == fast_cleared

    opt = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=True, minimize_draws=True)
    opt_seq, opt_cleared = opt.solve()
    opt_res = verify_tripeaks_solution(board_cards, stock_cards, opt_seq, wrap_enabled=True)
    assert opt_res["cleared"] == opt_cleared
    assert opt_cleared == fast_cleared
    if opt_cleared == 28:
        assert opt.min_stock_used <= fast.min_stock_used


def test_tripeaks_layered_bfs_on_shuffled_deal_and_state_cap():
    board_cards, stock_cards = _shuffled_deck(10)
    opt = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=True, minimize_draws=True)
    seq, cleared = opt.solve()
    assert cleared == 28
    res = verify_tripeaks_solution(board_cards, stock_cards, seq, wrap_enabled=True)
    assert res["cleared"] == 28
    assert res["draws"] == opt.min_stock_used

    capped = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=True, minimize_draws=True, max_states=50)
    seq_cap, cleared_cap = capped.solve()
    assert capped.state_limit_reached is True
    res_cap = verify_tripeaks_solution(board_cards, stock_cards, seq_cap, wrap_enabled=True)
    assert res_cap["cleared"] == cleared_cap
