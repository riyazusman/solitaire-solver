from solvers.freecell import FreecellSolver, FreecellState
from solvers.models import Card, DEFAULT_DECK_STR, parse_board_and_stock, parse_deck
from solvers.pyramid import PyramidBoard, PyramidSolver
from solvers.replay import replay_freecell, replay_pyramid, replay_tripeaks
from solvers.tripeaks import TriPeaksBoard, TriPeaksSolver
from tests.verifiers import verify_freecell_solution


def test_replay_tripeaks_matches_final_state():
    board_cards, stock_cards = parse_board_and_stock(DEFAULT_DECK_STR)
    solver = TriPeaksSolver(board_cards, stock_cards, TriPeaksBoard(), wrap_enabled=True)
    seq, cleared = solver.solve()
    assert cleared == 28

    snaps = replay_tripeaks(board_cards, stock_cards, seq)
    assert snaps[0].cleared == 0
    assert snaps[0].board_mask == (1 << 28) - 1
    assert snaps[-1].cleared == 28
    assert snaps[-1].board_mask == 0
    assert snaps[-1].draws == solver.min_stock_used


def test_replay_pyramid_matches_final_state():
    board_cards, stock_cards = parse_board_and_stock(DEFAULT_DECK_STR)
    solver = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1, max_states=500)
    seq, cleared = solver.solve()
    snaps = replay_pyramid(board_cards, stock_cards, seq)
    assert len(snaps) == len(seq) + 1
    assert snaps[0].cleared == 0
    assert snaps[-1].cleared == cleared


def test_replay_freecell_matches_final_state():
    deck = parse_deck(DEFAULT_DECK_STR)
    solver = FreecellSolver(FreecellState(deck))
    seq, solved, _ = solver.solve()
    assert solved
    snaps = replay_freecell(deck, seq)
    assert len(snaps) == len(seq) + 1
    assert sum(snaps[0].foundations.values()) == 0
    assert all(v == 13 for v in snaps[-1].foundations.values())


def test_pyramid_relaxed_waste_toggle_and_unlimited_passes():
    # Buried waste+waste endgame: disallowed in strict_waste=True, allowed in strict_waste=False
    board_cards = [Card("2S")] + [Card("3S")] * 27
    stock_cards = [Card("JD"), Card("6H"), Card("7H")]

    strict = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1, strict_waste=True)
    strict._best_moves = []
    strict.max_draws = 0
    assert not strict._dfs(0b1, (), None, (0, 1, 2), 0, 1, [], 27)

    relaxed = PyramidSolver(board_cards, stock_cards, PyramidBoard(), max_passes=1, strict_waste=False)
    relaxed._best_moves = []
    relaxed.max_draws = 0
    assert relaxed._dfs(0b1, (), None, (0, 1, 2), 0, 1, [], 27)

    # Unlimited passes (max_passes=None) solves multi-recycle endgame and terminates
    board2 = [Card("2S"), Card("9C"), Card("KS")] + [Card("3S")] * 25
    stock2 = [Card("JD"), Card("5D"), Card("4H")] + [Card("3D")] * 21
    unlim = PyramidSolver(board2, stock2, PyramidBoard(), max_passes=None)
    unlim._best_moves = []
    unlim.max_draws = float("inf")
    mask = (1 << 0) | (1 << 1) | (1 << 2)
    assert unlim._dfs(mask, (0, 1, 2), None, (), 0, 1, [], 25)
    moves = unlim._format_moves(unlim._best_moves)
    assert any("Pass 2/∞" in m for m in moves)


def test_freecell_allow_foundation_to_cascade_and_atomic_move_count():
    deck = parse_deck(DEFAULT_DECK_STR)

    # 1. Foundation -> Cascade generates valid pullback moves when enabled
    state = FreecellState(deck)
    state.foundations["S"] = 5  # 5S on foundation
    # Place 6H on Cascade 1 bottom so 5S can be pulled back onto 6H
    state.cascades[0].append(Card("6H"))
    solver_f2c = FreecellSolver(state, target_card=Card("7S"), allow_foundation_to_cascade=True)
    node_casc, node_fc, node_found = solver_f2c._to_int_node(state)
    moves = solver_f2c._generate_valid_moves([list(c) for c in node_casc], list(node_fc), list(node_found))
    assert any(m[0] == 5 for m in moves), "Expected Foundation -> Cascade pullback move"

    # 2. Atomic move count runs cleanly and verifies
    solver_atomic = FreecellSolver(FreecellState(deck), atomic_move_count=True)
    seq, solved, _ = solver_atomic.solve()
    assert solved
    assert solver_atomic.best_move_cost >= len(seq)
    verify_freecell_solution(deck, seq, expect_solved=True)
