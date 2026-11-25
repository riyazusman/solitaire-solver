"""Pure-Python move replay engine for step-by-step board snapshots.

Produces immutable board state snapshots for every step (0..N) of a TriPeaks,
Pyramid, or Freecell move sequence. Used by the interactive UI playback viewer.
"""

from dataclasses import dataclass
import re

from solvers.freecell import FreecellState
from solvers.models import Card

_SYMBOL_TO_SUIT = {
    "♠️": "S", "♠": "S",
    "♥️": "H", "♥": "H",
    "♦️": "D", "♦": "D",
    "♣️": "C", "♣": "C",
}

_CARD_TOKEN_RE = r"(?:10|[A2-9TJQK])(?:♠️|♥️|♦️|♣️|♠|♥|♦|♣)"


def parse_display_card(display_str: str) -> Card:
    s = display_str.strip()
    for sym, suit in _SYMBOL_TO_SUIT.items():
        if s.endswith(sym):
            return Card(f"{s[: -len(sym)]}{suit}")
    raise ValueError(f"Cannot parse displayed card token: {display_str!r}")


# ---------------------------------------------------------------------------
# TriPeaks Replay
# ---------------------------------------------------------------------------

_TP_ROW_OFFSETS = {1: 0, 2: 3, 3: 9, 4: 18}
_TP_START_RE = re.compile(rf"^Start: Waste pile is ({_CARD_TOKEN_RE})$")
_TP_PLAY_RE = re.compile(rf"^Play ({_CARD_TOKEN_RE}) from board \(Row (\d+), Position (\d+)\)$")
_TP_DRAW_RE = re.compile(rf"^Draw ({_CARD_TOKEN_RE}) from stock$")


@dataclass(frozen=True)
class TriPeaksSnapshot:
    step_num: int
    move_text: str
    board_mask: int
    waste: Card
    stock_remaining: int
    cleared: int
    draws: int


def replay_tripeaks(board_cards: list[Card], stock_cards: list[Card], sequence: list[str]) -> list[TriPeaksSnapshot]:
    board_mask = (1 << 28) - 1
    waste = stock_cards[0]
    stock_idx = 1
    cleared = 0
    draws = 0

    start_text = sequence[0] if sequence and _TP_START_RE.match(sequence[0]) else f"Start: Waste pile is {waste.display}"
    steps_to_replay = sequence[1:] if (sequence and _TP_START_RE.match(sequence[0])) else sequence

    snapshots = [
        TriPeaksSnapshot(
            step_num=0,
            move_text=start_text,
            board_mask=board_mask,
            waste=waste,
            stock_remaining=len(stock_cards) - stock_idx,
            cleared=cleared,
            draws=draws,
        )
    ]

    for step_num, step in enumerate(steps_to_replay, start=1):
        m = _TP_PLAY_RE.match(step)
        if m:
            idx = _TP_ROW_OFFSETS[int(m.group(2))] + (int(m.group(3)) - 1)
            board_mask &= ~(1 << idx)
            waste = board_cards[idx]
            cleared += 1
        elif _TP_DRAW_RE.match(step):
            waste = stock_cards[stock_idx]
            stock_idx += 1
            draws += 1
        snapshots.append(
            TriPeaksSnapshot(
                step_num=step_num,
                move_text=step,
                board_mask=board_mask,
                waste=waste,
                stock_remaining=len(stock_cards) - stock_idx,
                cleared=cleared,
                draws=draws,
            )
        )
    return snapshots


# ---------------------------------------------------------------------------
# Pyramid Replay
# ---------------------------------------------------------------------------

_PYR_ROW_OFFSETS = {1: 0, 2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21}
_PYR_BOARD_KING_RE = re.compile(rf"^Play ({_CARD_TOKEN_RE}) from board \(Row (\d+), Position (\d+)\)$")
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
_PYR_RECYCLE_RE = re.compile(r"^Recycle waste pile to stock \(Pass (\d+)/(.+)\)$")


@dataclass(frozen=True)
class PyramidSnapshot:
    step_num: int
    move_text: str
    board_mask: int
    drawn: Card | None
    waste_top: Card | None
    waste_count: int
    stock_remaining: int
    cleared: int
    draws: int
    passes: int


def replay_pyramid(board_cards: list[Card], stock_cards: list[Card], sequence: list[str]) -> list[PyramidSnapshot]:
    board_mask = (1 << 28) - 1
    stock = list(stock_cards)
    waste: list[Card] = []
    drawn: Card | None = None
    cleared = 0
    draws = 0
    passes = 1

    def snap(step_num: int, text: str) -> PyramidSnapshot:
        return PyramidSnapshot(
            step_num=step_num,
            move_text=text,
            board_mask=board_mask,
            drawn=drawn,
            waste_top=waste[-1] if waste else None,
            waste_count=len(waste),
            stock_remaining=len(stock),
            cleared=cleared,
            draws=draws,
            passes=passes,
        )

    snapshots = [snap(0, "Initial Pyramid Layout")]

    for step_num, step in enumerate(sequence, start=1):
        m = _PYR_BOARD_KING_RE.match(step)
        if m:
            idx = _PYR_ROW_OFFSETS[int(m.group(2))] + (int(m.group(3)) - 1)
            board_mask &= ~(1 << idx)
            cleared += 1
        elif (m := _PYR_BOARD_PAIR_RE.match(step)):
            idx1 = _PYR_ROW_OFFSETS[int(m.group(2))] + (int(m.group(3)) - 1)
            idx2 = _PYR_ROW_OFFSETS[int(m.group(5))] + (int(m.group(6)) - 1)
            board_mask &= ~(1 << idx1) & ~(1 << idx2)
            cleared += 2
        elif _PYR_DRAW_RE.match(step):
            if drawn is not None:
                waste.append(drawn)
            drawn = stock.pop(0)
            draws += 1
        elif _PYR_DRAW_KING_RE.match(step):
            if drawn is not None:
                waste.append(drawn)
            stock.pop(0)
            drawn = None
            draws += 1
        elif (m := _PYR_WASTE_KING_RE.match(step)):
            target = parse_display_card(m.group(1))
            if drawn is not None and drawn == target:
                drawn = None
            elif waste:
                waste.pop()
        elif (m := _PYR_DRAW_BOARD_PAIR_RE.match(step)):
            if drawn is not None:
                waste.append(drawn)
            stock.pop(0)
            drawn = None
            draws += 1
            idx = _PYR_ROW_OFFSETS[int(m.group(3))] + (int(m.group(4)) - 1)
            board_mask &= ~(1 << idx)
            cleared += 1
        elif (m := _PYR_BOARD_WASTE_PAIR_RE.match(step)):
            idx = _PYR_ROW_OFFSETS[int(m.group(2))] + (int(m.group(3)) - 1)
            target = parse_display_card(m.group(4))
            board_mask &= ~(1 << idx)
            if drawn is not None and drawn == target:
                drawn = None
            elif waste:
                waste.pop()
            cleared += 1
        elif _PYR_DRAW_WASTE_PAIR_RE.match(step):
            if drawn is not None:
                waste.append(drawn)
            stock.pop(0)
            drawn = None
            waste.pop()
            draws += 1
        elif (m := _PYR_WASTE_WASTE_PAIR_RE.match(step)):
            c_top = parse_display_card(m.group(1))
            if drawn is not None and drawn == c_top:
                drawn = None
                waste.pop()
            else:
                waste.pop()
                waste.pop()
        elif _PYR_RECYCLE_RE.match(step):
            if drawn is not None:
                waste.append(drawn)
                drawn = None
            stock = waste
            waste = []
            passes += 1

        snapshots.append(snap(step_num, step))
    return snapshots


# ---------------------------------------------------------------------------
# Freecell Replay
# ---------------------------------------------------------------------------

_FC_CASC_TO_FOUND_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to Foundation(?: \(auto\))?$")
_FC_FC_TO_FOUND_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Freecell to Foundation(?: \(auto\))?$")
_FC_CASC_TO_FC_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to Freecell$")
_FC_FC_TO_EMPTY_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Freecell to empty Cascade (\d+)$")
_FC_FC_TO_CASC_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Freecell to Cascade (\d+) on ({_CARD_TOKEN_RE})$")
_FC_FOUND_TO_EMPTY_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Foundation to empty Cascade (\d+)$")
_FC_FOUND_TO_CASC_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Foundation to Cascade (\d+) on ({_CARD_TOKEN_RE})$")
_FC_CASC_SINGLE_EMPTY_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to empty Cascade (\d+)$")
_FC_CASC_SINGLE_ON_RE = re.compile(rf"^Move ({_CARD_TOKEN_RE}) from Cascade (\d+) to Cascade (\d+) on ({_CARD_TOKEN_RE})$")
_FC_CASC_MULTI_EMPTY_RE = re.compile(
    rf"^Move (\d+) cards \(({_CARD_TOKEN_RE}) to ({_CARD_TOKEN_RE})\) from Cascade (\d+) to empty Cascade (\d+)$"
)
_FC_CASC_MULTI_ON_RE = re.compile(
    rf"^Move (\d+) cards \(({_CARD_TOKEN_RE}) to ({_CARD_TOKEN_RE})\) from Cascade (\d+) to Cascade (\d+) on ({_CARD_TOKEN_RE})$"
)


@dataclass(frozen=True)
class FreecellSnapshot:
    step_num: int
    move_text: str
    cascades: tuple[tuple[Card, ...], ...]
    freecells: tuple[Card, ...]
    foundations: dict[str, int]


def replay_freecell(deck_cards: list[Card], sequence: list[str]) -> list[FreecellSnapshot]:
    state = FreecellState(deck_cards)

    def snap(step_num: int, text: str) -> FreecellSnapshot:
        return FreecellSnapshot(
            step_num=step_num,
            move_text=text,
            cascades=tuple(tuple(c) for c in state.cascades),
            freecells=tuple(state.freecells),
            foundations=dict(state.foundations),
        )

    snapshots = [snap(0, "Initial Freecell Deal")]

    for step_num, step in enumerate(sequence, start=1):
        if (m := _FC_CASC_TO_FOUND_RE.match(step)):
            card = state.cascades[int(m.group(2)) - 1].pop()
            state.foundations[card.suit] += 1
        elif (m := _FC_FC_TO_FOUND_RE.match(step)):
            target = parse_display_card(m.group(1))
            idx = state.freecells.index(target)
            card = state.freecells.pop(idx)
            state.foundations[card.suit] += 1
        elif (m := _FC_CASC_TO_FC_RE.match(step)):
            card = state.cascades[int(m.group(2)) - 1].pop()
            state.freecells.append(card)
        elif (m := _FC_FC_TO_EMPTY_RE.match(step)) or (m := _FC_FC_TO_CASC_RE.match(step)):
            target = parse_display_card(m.group(1))
            dest = int(m.group(2)) - 1
            card = state.freecells.pop(state.freecells.index(target))
            state.cascades[dest].append(card)
        elif (m := _FC_FOUND_TO_EMPTY_RE.match(step)) or (m := _FC_FOUND_TO_CASC_RE.match(step)):
            card = parse_display_card(m.group(1))
            dest = int(m.group(2)) - 1
            state.foundations[card.suit] -= 1
            state.cascades[dest].append(card)
        elif (m := _FC_CASC_SINGLE_EMPTY_RE.match(step)) or (m := _FC_CASC_SINGLE_ON_RE.match(step)):
            src = int(m.group(2)) - 1
            dest = int(m.group(3)) - 1
            state.cascades[dest].append(state.cascades[src].pop())
        elif (m := _FC_CASC_MULTI_EMPTY_RE.match(step)) or (m := _FC_CASC_MULTI_ON_RE.match(step)):
            count = int(m.group(1))
            src = int(m.group(4)) - 1
            dest = int(m.group(5)) - 1
            moved = state.cascades[src][-count:]
            del state.cascades[src][-count:]
            state.cascades[dest].extend(moved)

        snapshots.append(snap(step_num, step))
    return snapshots
