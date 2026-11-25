import random
import pytest

from solvers.models import Card, STANDARD_DECK
from solvers.pyramid import PyramidBoard, PyramidSolver
from tests.verifiers import verify_pyramid_solution


def _shuffled_deck(seed: int):
    rng = random.Random(seed)
    cards = sorted(STANDARD_DECK)
    rng.shuffle(cards)
    parsed = [Card(c) for c in cards]
    return parsed[:28], parsed[28:]


def test_pyramid_board_geometry():
    board = PyramidBoard()
    assert board.get_initial_free_cards() == list(range(21, 28))
    assert board.get_row_and_position(0) == (1, 1)
    assert board.get_row_and_position(1) == (2, 1)
    assert board.get_row_and_position(3) == (3, 1)
    assert board.get_row_and_position(6) == (4, 1)
    assert board.get_row_and_position(10) == (5, 1)
    assert board.get_row_and_position(15) == (6, 1)
    assert board.get_row_and_position(21) == (7, 1)
    assert board.get_row_and_position(27) == (7, 7)


def test_pyramid_1_pass_seeded_deals_verify():
    for seed in (5,):
        board_cards, stock_cards = _shuffled_deck(seed)
        solver = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1)
        seq, cleared = solver.solve()
        result = verify_pyramid_solution(
            board_cards, stock_cards, seq, max_passes=1, strict_waste=True
        )
        assert result["cleared"] == cleared
        assert result["draws"] == solver.min_stock_used


def test_pyramid_minimize_draws_never_worse():
    board_cards, stock_cards = _shuffled_deck(5)
    fast = PyramidSolver(board_cards, stock_cards, PyramidBoard(), minimize_draws=False, max_passes=1)
    _, fast_cleared = fast.solve()

    opt = PyramidSolver(board_cards, stock_cards, PyramidBoard(), minimize_draws=True, max_passes=1)
    opt_seq, opt_cleared = opt.solve()
    res = verify_pyramid_solution(board_cards, stock_cards, opt_seq, max_passes=1, strict_waste=True)
    assert res["cleared"] == opt_cleared
    assert opt_cleared == fast_cleared
    if opt_cleared == 28:
        assert opt.min_stock_used <= fast.min_stock_used


def test_pyramid_disallows_buried_waste_waste_pairing():
    # Minimal endgame state: only board idx 0 (2S) remains.
    # Waste has (JD, 6H, 7H), drawn_idx=None (drawn card KH was already played & removed).
    # Under standard Pyramid rules, 7H and 6H are both in the waste pile and CANNOT be paired
    # to uncover JD for 2S.
    board_cards = [Card("2S")] + [Card("3S")] * 27
    stock_cards = [Card("JD"), Card("6H"), Card("7H")]
    solver = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1)
    solver._best_moves = []
    solver.visited_states.clear()
    solver.max_draws = 0

    won = solver._dfs(0b1, (), None, (0, 1, 2), 0, 1, [], 27)
    assert not won, f"Solver illegally paired buried waste cards: {solver._format_moves(solver._best_moves)}"


def test_pyramid_allows_active_drawn_card_paired_with_waste_top_after_board_move():
    # Board has idx 0 (2S) covered by idx 1 (KS) and idx 2 (QS) -> actually idx 0 uncovered, idx 1 uncovered (2 cards):
    # Suppose board bits 0 (2S) and 21 (KS) are active (both uncovered when bits 1..20,22..27 are 0).
    # Drawn slot has 7H, waste has (JD, 6H).
    # Playing board KS must NOT bury 7H; afterwards 7H (drawn) + 6H (waste top) pair legally,
    # uncovering JD to pair with 2S.
    board_cards = [Card("2S")] + [Card("3S")] * 20 + [Card("KS")] + [Card("3S")] * 6
    stock_cards = [Card("JD"), Card("6H"), Card("7H")]
    solver = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1)
    solver._best_moves = []
    solver.visited_states.clear()
    solver.max_draws = 0

    mask = (1 << 0) | (1 << 21)
    won = solver._dfs(mask, (), 2, (0, 1), 0, 1, [], 26)
    assert won
    moves = solver._format_moves(solver._best_moves)
    assert any("7♥️" in m and "6♥️" in m for m in moves)


def test_pyramid_3_pass_recycle_strict_verification():
    # Minimal 2-pass endgame: board has only idx 0 (2S).
    # Stock starts with (3H, JD) -> pass 1 draws 3H then JD (pairs with 2S on pass 1),
    # OR put JD in waste and force a recycle to pass 2 before pairing!
    # Here: board has idx 0 (2S) covered by idx 1 (4H) and idx 2 (9S).
    # Wait: even simpler — stock has (JD, 4H), board has idx 0 (2S) covered by idx 1 (9C) and idx 2 (KS).
    # On pass 1: draw JD (cannot pair 2S yet, 2S covered), draw 4H -> pair 4H with 9C (uncovers 2S after KS),
    # play KS -> now 2S is free, JD is on waste top! Wait, JD IS on waste top already.
    # To force a recycle: stock=(JD, 5D, 4H), board has idx 0 (2S) covered by idx 1 (9C) and idx 2 (KS).
    # Pass 1: draw JD -> draw 5D (buries JD!) -> draw 4H (pairs 9C) -> play KS -> 2S free, JD buried under 5D!
    # Must recycle (5D, JD? No, JD is first in waste -> recycled stock is (JD, 5D)) -> Pass 2 draw JD -> pair 2S!
    board_cards = [Card("2S"), Card("9C"), Card("KS")] + [Card("3S")] * 25
    stock_cards = [Card("JD"), Card("5D"), Card("4H")] + [Card("3D")] * 21
    solver = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=2)
    solver._best_moves = []
    solver.visited_states.clear()
    solver.max_draws = 6
    mask = (1 << 0) | (1 << 1) | (1 << 2)
    won = solver._dfs(mask, (0, 1, 2), None, (), 0, 1, [], 25)
    assert won
    moves = solver._format_moves(solver._best_moves)
    assert any("Recycle waste pile to stock (Pass 2/2)" in m for m in moves)


def test_pyramid_state_limit_cap_returns_valid_partial():
    board_cards, stock_cards = _shuffled_deck(5)
    solver = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1, max_states=50)
    seq, cleared = solver.solve()
    assert solver.state_limit_reached is True
    res = verify_pyramid_solution(board_cards, stock_cards, seq, max_passes=1, strict_waste=True)
    assert res["cleared"] == cleared
