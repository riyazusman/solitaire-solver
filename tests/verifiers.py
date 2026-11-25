"""Independent move-legality verifiers for TriPeaks, Pyramid, and Freecell.

Each verifier replays a solver's emitted move sequence from the starting deck
against the official game rules and raises AssertionError on any illegal move.
"""

import re

from solvers.models import Card
from solvers.freecell import FreecellState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYMBOL_TO_SUIT = {
    "♠️": "S", "♠": "S",
    "♥️": "H", "♥": "H",
    "♦️": "D", "♦": "D",
    "♣️": "C", "♣": "C",
}

_CARD_TOKEN_RE = r"(?:10|[A2-9TJQK])(?:♠️|♥️|♦️|♣️|♠|♥|♦|♣)"


def parse_display_card(display_str: str) -> Card:
    """Convert a Card.display string like '10♠️' or 'A♥️' back into a Card."""
    s = display_str.strip()
    for sym, suit in _SYMBOL_TO_SUIT.items():
        if s.endswith(sym):
            rank = s[: -len(sym)]
            return Card(f"{rank}{suit}")
    raise AssertionError(f"Cannot parse displayed card token: {display_str!r}")


def _same_card(a: Card, b: Card) -> bool:
    return a.card_str == b.card_str


# ---------------------------------------------------------------------------
# TriPeaks Verifier
# ---------------------------------------------------------------------------

_TRIPEAKS_COVERED_BY = {
    0: (3, 4), 1: (5, 6), 2: (7, 8),
    3: (9, 10), 4: (10, 11), 5: (12, 13), 6: (13, 14), 7: (15, 16), 8: (16, 17),
    9: (18, 19), 10: (19, 20), 11: (20, 21),
    12: (21, 22), 13: (22, 23), 14: (23, 24),
    15: (24, 25), 16: (25, 26), 17: (26, 27),
}

_TRIPEAKS_ROW_OFFSETS = {1: 0, 2: 3, 3: 9, 4: 18}
_TRIPEAKS_ROW_LENGTHS = {1: 3, 2: 6, 3: 9, 4: 10}

_TP_START_RE = re.compile(rf"^Start: Waste pile is ({_CARD_TOKEN_RE})$")
_TP_PLAY_RE = re.compile(
    rf"^Play ({_CARD_TOKEN_RE}) from board \(Row (\d+), Position (\d+)\)$"
)
_TP_DRAW_RE = re.compile(rf"^Draw ({_CARD_TOKEN_RE}) from stock$")


def _tripeaks_idx(row: int, pos: int) -> int:
    assert row in _TRIPEAKS_ROW_OFFSETS, f"Invalid TriPeaks row: {row}"
    assert 1 <= pos <= _TRIPEAKS_ROW_LENGTHS[row], f"Invalid TriPeaks pos {pos} in row {row}"
    return _TRIPEAKS_ROW_OFFSETS[row] + (pos - 1)


def verify_tripeaks_solution(board_cards, stock_cards, sequence, wrap_enabled=True):
    """Replay a TriPeaks move sequence and return final metrics."""
    assert len(board_cards) == 28
    assert len(stock_cards) == 24

    board_mask = (1 << 28) - 1
    waste = stock_cards[0]
    stock_idx = 1
    cleared = 0
    draws = 0

    if not sequence:
        return {"cleared": 0, "draws": 0, "waste": waste}

    m0 = _TP_START_RE.match(sequence[0])
    assert m0, f"Unexpected TriPeaks start step: {sequence[0]!r}"
    assert _same_card(parse_display_card(m0.group(1)), waste), (
        f"Start waste {m0.group(1)} does not match stock_cards[0] ({waste.display})"
    )

    for step_num, step in enumerate(sequence[1:], start=1):
        m_play = _TP_PLAY_RE.match(step)
        if m_play:
            card = parse_display_card(m_play.group(1))
            idx = _tripeaks_idx(int(m_play.group(2)), int(m_play.group(3)))
            assert board_mask & (1 << idx), f"Step {step_num}: board idx {idx} already cleared ({step})"
            assert _same_card(board_cards[idx], card), (
                f"Step {step_num}: expected {board_cards[idx].display} at idx {idx}, got {card.display}"
            )
            for cov in _TRIPEAKS_COVERED_BY.get(idx, ()):
                assert not (board_mask & (1 << cov)), (
                    f"Step {step_num}: card {card.display} (idx {idx}) is still covered by idx {cov}"
                )
            assert waste.is_adjacent(board_cards[idx], wrap_enabled=wrap_enabled), (
                f"Step {step_num}: {card.display} is not adjacent to waste {waste.display} "
                f"(wrap_enabled={wrap_enabled})"
            )
            board_mask &= ~(1 << idx)
            waste = board_cards[idx]
            cleared += 1
            continue

        m_draw = _TP_DRAW_RE.match(step)
        if m_draw:
            card = parse_display_card(m_draw.group(1))
            assert stock_idx < len(stock_cards), f"Step {step_num}: stock exhausted ({step})"
            assert _same_card(stock_cards[stock_idx], card), (
                f"Step {step_num}: expected drawn card {stock_cards[stock_idx].display}, got {card.display}"
            )
            waste = stock_cards[stock_idx]
            stock_idx += 1
            draws += 1
            continue

        raise AssertionError(f"Step {step_num}: unrecognized TriPeaks move format: {step!r}")

    return {"cleared": cleared, "draws": draws, "waste": waste}


# ---------------------------------------------------------------------------
# Pyramid Verifier
# ---------------------------------------------------------------------------

_PYRAMID_COVERED_BY = {
    0: (1, 2),
    1: (3, 4), 2: (4, 5),
    3: (6, 7), 4: (7, 8), 5: (8, 9),
    6: (10, 11), 7: (11, 12), 8: (12, 13), 9: (13, 14),
    10: (15, 16), 11: (16, 17), 12: (17, 18), 13: (18, 19), 14: (19, 20),
    15: (21, 22), 16: (22, 23), 17: (23, 24), 18: (24, 25), 19: (25, 26), 20: (26, 27),
}

_PYR_ROW_OFFSETS = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21}

_PYR_BOARD_KING_RE = re.compile(
    rf"^Play ({_CARD_TOKEN_RE}) from board \(Row (\d+), Position (\d+)\)$"
)
_PYR_BOARD_PAIR_RE = re.compile(
    rf"^Pair ({_CARD_TOKEN_RE}) \(Row (\d+), Pos (\d+)\) and ({_CARD_TOKEN_RE}) \(Row (\d+), Pos (\d+)\)$"
)
_PYR_DRAW_RE = re.compile(rf"^Draw ({_CARD_TOKEN_RE}) from stock$")
_PYR_DRAW_KING_RE = re.compile(rf"^Draw and immediately play ({_CARD_TOKEN_RE})$")
_PYR_WASTE_KING_RE = re.compile(rf"^Play ({_CARD_TOKEN_RE}) from waste$")
_PYR_DRAW_BOARD_PAIR_RE = re.compile(
    rf"^Draw ({_CARD_TOKEN_RE}) and immediately pair with ({_CARD_TOKEN_RE}) from board \(Row (\d+), Pos (\d+)\)$"
)
_PYR_BOARD_WASTE_PAIR_RE = re.compile(
    rf"^Pair ({_CARD_TOKEN_RE}) from board \(Row (\d+), Pos (\d+)\) and ({_CARD_TOKEN_RE}) from waste$"
)
_PYR_DRAW_WASTE_PAIR_RE = re.compile(
    rf"^Draw ({_CARD_TOKEN_RE}) and immediately pair with ({_CARD_TOKEN_RE}) in waste$"
)
_PYR_WASTE_WASTE_PAIR_RE = re.compile(
    rf"^Pair ({_CARD_TOKEN_RE}) and ({_CARD_TOKEN_RE}) in waste$"
)
_PYR_RECYCLE_RE = re.compile(r"^Recycle waste pile to stock \(Pass (\d+)/(\d+|∞)\)$")


def _pyramid_idx(row: int, pos: int) -> int:
    assert 1 <= row <= 7, f"Invalid Pyramid row: {row}"
    assert 1 <= pos <= row, f"Invalid Pyramid pos {pos} in row {row}"
    return _PYR_ROW_OFFSETS[row] + (pos - 1)


def verify_pyramid_solution(
    board_cards, stock_cards, sequence, max_passes=1, strict_waste=True
):
    """Replay a Pyramid move sequence and return final metrics.

    When strict_waste=True (standard Pyramid rules), only the active drawn card
    and the top card of the waste pile can be paired together; two buried cards
    in the waste pile cannot be paired after the drawn card has been removed.
    """
    assert len(board_cards) == 28
    assert len(stock_cards) == 24

    board_mask = (1 << 28) - 1
    stock = list(stock_cards)
    waste = []
    drawn = None
    passes = 1
    draws = 0
    cleared = 0

    def assert_board_free(idx: int, expected: Card, step_num: int):
        assert board_mask & (1 << idx), f"Step {step_num}: board idx {idx} already cleared"
        assert _same_card(board_cards[idx], expected), (
            f"Step {step_num}: expected {board_cards[idx].display} at idx {idx}, got {expected.display}"
        )
        for cov in _PYRAMID_COVERED_BY.get(idx, ()):
            assert not (board_mask & (1 << cov)), (
                f"Step {step_num}: card {expected.display} (idx {idx}) still covered by idx {cov}"
            )

    def do_draw(expected: Card, step_num: int) -> Card:
        nonlocal drawn, draws
        assert stock, f"Step {step_num}: cannot draw from empty stock"
        if drawn is not None:
            waste.append(drawn)
        card = stock.pop(0)
        assert _same_card(card, expected), (
            f"Step {step_num}: expected drawn card {card.display}, got {expected.display}"
        )
        drawn = card
        draws += 1
        return card

    def consume_top_waste_or_drawn(expected: Card, step_num: int) -> Card:
        nonlocal drawn
        if drawn is not None and _same_card(drawn, expected):
            card = drawn
            drawn = None
            return card
        if waste and _same_card(waste[-1], expected):
            # In strict Pyramid, if drawn is present it sits in the draw slot
            # while waste[-1] is also exposed on top of the waste pile.
            return waste.pop()
        raise AssertionError(
            f"Step {step_num}: {expected.display} is not available in draw/waste "
            f"(drawn={drawn}, waste_top={waste[-1] if waste else None})"
        )

    for step_num, step in enumerate(sequence, start=1):
        m = _PYR_BOARD_KING_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            idx = _pyramid_idx(int(m.group(2)), int(m.group(3)))
            assert_board_free(idx, card, step_num)
            assert card.is_king(), f"Step {step_num}: {card.display} is not a King"
            board_mask &= ~(1 << idx)
            cleared += 1
            continue

        m = _PYR_BOARD_PAIR_RE.match(step)
        if m:
            c1 = parse_display_card(m.group(1))
            idx1 = _pyramid_idx(int(m.group(2)), int(m.group(3)))
            c2 = parse_display_card(m.group(4))
            idx2 = _pyramid_idx(int(m.group(5)), int(m.group(6)))
            assert idx1 != idx2, f"Step {step_num}: cannot pair a card with itself"
            assert_board_free(idx1, c1, step_num)
            assert_board_free(idx2, c2, step_num)
            assert c1.is_pyramid_pair(c2), f"Step {step_num}: {c1.display} + {c2.display} != 13"
            board_mask &= ~(1 << idx1) & ~(1 << idx2)
            cleared += 2
            continue

        m = _PYR_DRAW_RE.match(step)
        if m:
            do_draw(parse_display_card(m.group(1)), step_num)
            continue

        m = _PYR_DRAW_KING_RE.match(step)
        if m:
            card = do_draw(parse_display_card(m.group(1)), step_num)
            assert card.is_king(), f"Step {step_num}: {card.display} is not a King"
            drawn = None
            continue

        m = _PYR_WASTE_KING_RE.match(step)
        if m:
            card = consume_top_waste_or_drawn(parse_display_card(m.group(1)), step_num)
            assert card.is_king(), f"Step {step_num}: {card.display} is not a King"
            continue

        m = _PYR_DRAW_BOARD_PAIR_RE.match(step)
        if m:
            c_draw = do_draw(parse_display_card(m.group(1)), step_num)
            c_board = parse_display_card(m.group(2))
            idx = _pyramid_idx(int(m.group(3)), int(m.group(4)))
            assert_board_free(idx, c_board, step_num)
            assert c_draw.is_pyramid_pair(c_board), (
                f"Step {step_num}: {c_draw.display} + {c_board.display} != 13"
            )
            board_mask &= ~(1 << idx)
            drawn = None
            cleared += 1
            continue

        m = _PYR_BOARD_WASTE_PAIR_RE.match(step)
        if m:
            c_board = parse_display_card(m.group(1))
            idx = _pyramid_idx(int(m.group(2)), int(m.group(3)))
            c_waste = parse_display_card(m.group(4))
            assert_board_free(idx, c_board, step_num)
            actual_waste = consume_top_waste_or_drawn(c_waste, step_num)
            assert c_board.is_pyramid_pair(actual_waste), (
                f"Step {step_num}: {c_board.display} + {actual_waste.display} != 13"
            )
            board_mask &= ~(1 << idx)
            cleared += 1
            continue

        m = _PYR_DRAW_WASTE_PAIR_RE.match(step)
        if m:
            c_draw = do_draw(parse_display_card(m.group(1)), step_num)
            c_sub = parse_display_card(m.group(2))
            assert waste and _same_card(waste[-1], c_sub), (
                f"Step {step_num}: {c_sub.display} is not on top of waste"
            )
            assert c_draw.is_pyramid_pair(waste[-1]), (
                f"Step {step_num}: {c_draw.display} + {waste[-1].display} != 13"
            )
            waste.pop()
            drawn = None
            continue

        m = _PYR_WASTE_WASTE_PAIR_RE.match(step)
        if m:
            c_top = parse_display_card(m.group(1))
            c_sub = parse_display_card(m.group(2))
            if strict_waste:
                assert drawn is not None and _same_card(drawn, c_top), (
                    f"Step {step_num}: illegal buried waste+waste pairing ({step}); "
                    f"top card {c_top.display} is not the active drawn card (drawn={drawn})"
                )
                assert waste and _same_card(waste[-1], c_sub), (
                    f"Step {step_num}: {c_sub.display} is not on top of waste"
                )
                assert drawn.is_pyramid_pair(waste[-1]), (
                    f"Step {step_num}: {drawn.display} + {waste[-1].display} != 13"
                )
                drawn = None
                waste.pop()
            else:
                if drawn is not None and _same_card(drawn, c_top):
                    assert waste and _same_card(waste[-1], c_sub)
                    assert drawn.is_pyramid_pair(waste[-1])
                    drawn = None
                    waste.pop()
                else:
                    assert len(waste) >= 2 and _same_card(waste[-1], c_top) and _same_card(waste[-2], c_sub)
                    assert waste[-1].is_pyramid_pair(waste[-2])
                    waste.pop()
                    waste.pop()
            continue

        m = _PYR_RECYCLE_RE.match(step)
        if m:
            next_pass = int(m.group(1))
            reported_max = None if m.group(2) == "∞" else int(m.group(2))
            assert reported_max == max_passes
            assert not stock, f"Step {step_num}: cannot recycle while stock has {len(stock)} cards"
            assert max_passes is None or passes < max_passes, f"Step {step_num}: exceeded max_passes={max_passes}"
            if drawn is not None:
                waste.append(drawn)
                drawn = None
            assert waste, f"Step {step_num}: cannot recycle empty waste pile"
            passes += 1
            assert next_pass == passes
            stock = waste
            waste = []
            continue

        raise AssertionError(f"Step {step_num}: unrecognized Pyramid move format: {step!r}")

    return {"cleared": cleared, "draws": draws, "passes": passes}


# ---------------------------------------------------------------------------
# Freecell Verifier
# ---------------------------------------------------------------------------

_FC_CASC_TO_FOUND_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to Foundation(?: \(auto\))?$"
)
_FC_FC_TO_FOUND_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Freecell to Foundation(?: \(auto\))?$"
)
_FC_CASC_TO_FC_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to Freecell$"
)
_FC_FC_TO_EMPTY_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Freecell to empty Cascade (\d+)$"
)
_FC_FC_TO_CASC_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Freecell to Cascade (\d+) on ({_CARD_TOKEN_RE})$"
)
_FC_FOUND_TO_EMPTY_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Foundation to empty Cascade (\d+)$"
)
_FC_FOUND_TO_CASC_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Foundation to Cascade (\d+) on ({_CARD_TOKEN_RE})$"
)
_FC_CASC_SINGLE_EMPTY_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to empty Cascade (\d+)$"
)
_FC_CASC_SINGLE_ON_RE = re.compile(
    rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to Cascade (\d+) on ({_CARD_TOKEN_RE})$"
)
_FC_CASC_MULTI_EMPTY_RE = re.compile(
    rf"^Move (\d+) cards \(({_CARD_TOKEN_RE}) to ({_CARD_TOKEN_RE})\) from Cascade (\d+) to empty Cascade (\d+)$"
)
_FC_CASC_MULTI_ON_RE = re.compile(
    rf"^Move (\d+) cards \(({_CARD_TOKEN_RE}) to ({_CARD_TOKEN_RE})\) from Cascade (\d+) to Cascade (\d+) on ({_CARD_TOKEN_RE})$"
)


def _assert_valid_run(cards, step_num: int):
    for a, b in zip(cards, cards[1:]):
        assert a.color != b.color and a.value == b.value + 1, (
            f"Step {step_num}: invalid alternating descending run {[c.display for c in cards]}"
        )


def verify_freecell_solution(
    deck_cards,
    sequence,
    target_card=None,
    target_ranks=None,
    rank_count=4,
    move_limit=None,
    expect_solved=True,
):
    """Replay a Freecell move sequence and verify every rule & goal."""
    state = FreecellState(deck_cards)

    for step_num, step in enumerate(sequence, start=1):
        free_cells = 4 - len(state.freecells)
        empty_cols = sum(1 for c in state.cascades if not c)

        m = _FC_CASC_TO_FOUND_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            src = int(m.group(2)) - 1
            assert 0 <= src < 8 and state.cascades[src], f"Step {step_num}: empty cascade {src + 1}"
            top = state.cascades[src].pop()
            assert _same_card(top, card), f"Step {step_num}: expected {card.display}, found {top.display}"
            assert state.foundations[card.suit] == card.value - 1, (
                f"Step {step_num}: cannot place {card.display} on foundation {state.foundations[card.suit]}"
            )
            state.foundations[card.suit] += 1
            continue

        m = _FC_FC_TO_FOUND_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            idx = next((i for i, c in enumerate(state.freecells) if _same_card(c, card)), None)
            assert idx is not None, f"Step {step_num}: {card.display} not in freecells"
            state.freecells.pop(idx)
            assert state.foundations[card.suit] == card.value - 1, (
                f"Step {step_num}: cannot place {card.display} on foundation {state.foundations[card.suit]}"
            )
            state.foundations[card.suit] += 1
            continue

        m = _FC_CASC_TO_FC_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            src = int(m.group(2)) - 1
            assert free_cells > 0, f"Step {step_num}: no empty freecell available"
            assert 0 <= src < 8 and state.cascades[src], f"Step {step_num}: empty cascade {src + 1}"
            top = state.cascades[src].pop()
            assert _same_card(top, card), f"Step {step_num}: expected {card.display}, found {top.display}"
            state.freecells.append(top)
            continue

        m = _FC_FC_TO_EMPTY_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            dest = int(m.group(2)) - 1
            assert 0 <= dest < 8 and not state.cascades[dest], f"Step {step_num}: cascade {dest + 1} not empty"
            idx = next((i for i, c in enumerate(state.freecells) if _same_card(c, card)), None)
            assert idx is not None, f"Step {step_num}: {card.display} not in freecells"
            state.cascades[dest].append(state.freecells.pop(idx))
            continue

        m = _FC_FC_TO_CASC_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            dest = int(m.group(2)) - 1
            target = parse_display_card(m.group(3))
            assert 0 <= dest < 8 and state.cascades[dest], f"Step {step_num}: cascade {dest + 1} is empty"
            dest_top = state.cascades[dest][-1]
            assert _same_card(dest_top, target), (
                f"Step {step_num}: expected {target.display} on cascade {dest + 1}, got {dest_top.display}"
            )
            assert dest_top.color != card.color and dest_top.value == card.value + 1, (
                f"Step {step_num}: cannot place {card.display} on {dest_top.display}"
            )
            idx = next((i for i, c in enumerate(state.freecells) if _same_card(c, card)), None)
            assert idx is not None, f"Step {step_num}: {card.display} not in freecells"
            state.cascades[dest].append(state.freecells.pop(idx))
            continue

        m = _FC_FOUND_TO_EMPTY_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            dest = int(m.group(2)) - 1
            assert 0 <= dest < 8 and not state.cascades[dest], f"Step {step_num}: cascade {dest + 1} not empty"
            assert state.foundations[card.suit] == card.value, (
                f"Step {step_num}: {card.display} is not top of foundation"
            )
            state.foundations[card.suit] -= 1
            state.cascades[dest].append(card)
            continue

        m = _FC_FOUND_TO_CASC_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            dest = int(m.group(2)) - 1
            target = parse_display_card(m.group(3))
            assert 0 <= dest < 8 and state.cascades[dest], f"Step {step_num}: cascade {dest + 1} is empty"
            assert state.foundations[card.suit] == card.value, (
                f"Step {step_num}: {card.display} is not top of foundation"
            )
            dest_top = state.cascades[dest][-1]
            assert _same_card(dest_top, target)
            assert dest_top.color != card.color and dest_top.value == card.value + 1
            state.foundations[card.suit] -= 1
            state.cascades[dest].append(card)
            continue

        m = _FC_CASC_SINGLE_EMPTY_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            src = int(m.group(2)) - 1
            dest = int(m.group(3)) - 1
            assert src != dest and not state.cascades[dest], f"Step {step_num}: invalid empty target"
            top = state.cascades[src].pop()
            assert _same_card(top, card), f"Step {step_num}: expected {card.display}, got {top.display}"
            state.cascades[dest].append(top)
            continue

        m = _FC_CASC_SINGLE_ON_RE.match(step)
        if m:
            card = parse_display_card(m.group(1))
            src = int(m.group(2)) - 1
            dest = int(m.group(3)) - 1
            target = parse_display_card(m.group(4))
            assert src != dest and state.cascades[dest], f"Step {step_num}: invalid cascade target"
            dest_top = state.cascades[dest][-1]
            assert _same_card(dest_top, target)
            top = state.cascades[src].pop()
            assert _same_card(top, card)
            assert dest_top.color != top.color and dest_top.value == top.value + 1, (
                f"Step {step_num}: cannot place {top.display} on {dest_top.display}"
            )
            state.cascades[dest].append(top)
            continue

        m = _FC_CASC_MULTI_EMPTY_RE.match(step)
        if m:
            count = int(m.group(1))
            top_c = parse_display_card(m.group(2))
            bot_c = parse_display_card(m.group(3))
            src = int(m.group(4)) - 1
            dest = int(m.group(5)) - 1
            assert src != dest and not state.cascades[dest], f"Step {step_num}: target not empty"
            capacity = (free_cells + 1) * (2 ** (empty_cols - 1))
            assert count <= capacity, f"Step {step_num}: supermove of {count} exceeds empty capacity {capacity}"
            moved = state.cascades[src][-count:]
            assert len(moved) == count and _same_card(moved[0], top_c) and _same_card(moved[-1], bot_c)
            _assert_valid_run(moved, step_num)
            del state.cascades[src][-count:]
            state.cascades[dest].extend(moved)
            continue

        m = _FC_CASC_MULTI_ON_RE.match(step)
        if m:
            count = int(m.group(1))
            top_c = parse_display_card(m.group(2))
            bot_c = parse_display_card(m.group(3))
            src = int(m.group(4)) - 1
            dest = int(m.group(5)) - 1
            target = parse_display_card(m.group(6))
            assert src != dest and state.cascades[dest], f"Step {step_num}: target is empty"
            capacity = (free_cells + 1) * (2 ** empty_cols)
            assert count <= capacity, f"Step {step_num}: supermove of {count} exceeds capacity {capacity}"
            dest_top = state.cascades[dest][-1]
            assert _same_card(dest_top, target)
            moved = state.cascades[src][-count:]
            assert len(moved) == count and _same_card(moved[0], top_c) and _same_card(moved[-1], bot_c)
            _assert_valid_run(moved, step_num)
            assert dest_top.color != moved[0].color and dest_top.value == moved[0].value + 1
            del state.cascades[src][-count:]
            state.cascades[dest].extend(moved)
            continue

        raise AssertionError(f"Step {step_num}: unrecognized Freecell move format: {step!r}")

    if target_card is not None:
        goal_met = state.foundations[target_card.suit] >= target_card.value
    elif target_ranks:
        goal_met = all(
            sum(1 for h in state.foundations.values() if h >= r) >= rank_count
            for r in target_ranks
        )
    else:
        goal_met = state.is_solved()

    if expect_solved:
        assert goal_met, f"Goal not satisfied after replay. Foundations={state.foundations}"
        if move_limit is not None:
            assert len(sequence) <= move_limit, (
                f"Move count {len(sequence)} exceeds move_limit={move_limit}"
            )

    return {
        "foundations": dict(state.foundations),
        "moves": len(sequence),
        "goal_met": goal_met,
    }
