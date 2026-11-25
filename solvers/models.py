from collections import Counter
from dataclasses import dataclass
import re


class Card:
    __slots__ = ("card_str", "rank", "suit", "value", "is_red", "color", "int_id", "display")

    VALUES = {
        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13,
    }
    VALID_SUITS = {"S", "H", "D", "C"}
    SUIT_INDEX = {"S": 0, "H": 1, "D": 2, "C": 3}
    SUIT_SYMBOLS = {"S": "♠️", "H": "♥️", "D": "♦️", "C": "♣️"}

    def __init__(self, card_str: str):
        raw_str = card_str.strip().upper()
        if raw_str.startswith("10"):
            raw_str = "T" + raw_str[2:]

        if len(raw_str) != 2:
            raise ValueError(
                f"Invalid card format '{card_str}'. Expected rank and suit (e.g., 'AH')."
            )

        rank = raw_str[0]
        suit = raw_str[1]

        if rank not in self.VALUES:
            raise ValueError(f"Invalid rank '{rank}' in card '{card_str}'.")
        if suit not in self.VALID_SUITS:
            raise ValueError(
                f"Invalid suit '{suit}' in card '{card_str}'. Valid suits are S, H, D, C."
            )

        self.card_str = raw_str
        self.rank = rank
        self.suit = suit
        self.value = self.VALUES[rank]
        self.is_red = suit in ("H", "D")
        self.color = "RED" if self.is_red else "BLACK"
        self.int_id = self.SUIT_INDEX[suit] * 13 + (self.value - 1)
        display_rank = "10" if rank == "T" else rank
        self.display = f"{display_rank}{self.SUIT_SYMBOLS[suit]}"

    def is_adjacent(self, other_card: "Card", wrap_enabled: bool = True) -> bool:
        diff = abs(self.value - other_card.value)
        if diff == 1:
            return True
        if wrap_enabled and diff == 12:
            return True
        return False

    def is_pyramid_pair(self, other_card: "Card") -> bool:
        return self.value + other_card.value == 13

    def is_king(self) -> bool:
        return self.value == 13

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.int_id == other.int_id

    def __hash__(self) -> int:
        return hash(self.int_id)

    def __str__(self) -> str:
        return self.card_str

    def __repr__(self) -> str:
        return f"Card('{self.card_str}')"


STANDARD_DECK = frozenset(
    f"{rank}{suit}" for rank in Card.VALUES for suit in Card.VALID_SUITS
)

DEFAULT_DECK_STR = (
    "AS, 2S, 3S, 4S, 5S, 6S, 7S, 8S, 9S, TS, JS, QS, KS, "
    "AH, 2H, 3H, 4H, 5H, 6H, 7H, 8H, 9H, TH, JH, QH, KH, "
    "AD, 2D, 3D, 4D, 5D, 6D, 7D, 8D, 9D, TD, JD, QD, KD, "
    "AC, 2C, 3C, 4C, 5C, 6C, 7C, 8C, 9C, TC, JC, QC, KC"
)

_TOKEN_SPLIT_RE = re.compile(r"[\s,;]+")


def tokenize_deck_input(input_str: str) -> list[str]:
    """Split card input on commas, whitespace, newlines, or semicolons."""
    stripped = input_str.strip()
    if not stripped:
        return []
    return [tok for tok in _TOKEN_SPLIT_RE.split(stripped) if tok]


@dataclass(frozen=True)
class DeckAnalysis:
    total_entries: int
    valid_unique: int
    cards: tuple[Card, ...]
    missing: tuple[str, ...]
    invalids: tuple[str, ...]
    duplicates: tuple[tuple[str, int], ...]

    @property
    def is_valid(self) -> bool:
        return (
            self.total_entries == 52
            and self.valid_unique == 52
            and not self.invalids
            and not self.duplicates
            and not self.missing
        )

    @property
    def duplicate_labels(self) -> list[str]:
        return [
            f"{card} (x{count})" if count > 2 else card
            for card, count in self.duplicates
        ]


def analyze_deck(input_str: str) -> DeckAnalysis:
    """Parse and validate a raw deck string, returning structured status details."""
    raw_cards = tokenize_deck_input(input_str)
    parsed_cards: list[Card] = []
    counts: Counter[str] = Counter()
    invalids: list[str] = []

    for token in raw_cards:
        try:
            card = Card(token)
        except ValueError:
            invalids.append(token)
            continue
        parsed_cards.append(card)
        counts[card.card_str] += 1

    duplicates = tuple((card_str, count) for card_str, count in counts.items() if count > 1)
    missing = tuple(sorted(STANDARD_DECK - counts.keys()))
    return DeckAnalysis(
        total_entries=len(raw_cards),
        valid_unique=len(counts),
        cards=tuple(parsed_cards),
        missing=missing,
        invalids=tuple(invalids),
        duplicates=duplicates,
    )


def parse_deck(input_str: str) -> list[Card]:
    """Parse and validate a full 52-card deck, raising ValueError with details on failure."""
    analysis = analyze_deck(input_str)
    if analysis.is_valid:
        return list(analysis.cards)

    errors = []
    if analysis.total_entries != 52:
        errors.append(f"Expected exactly 52 cards, got {analysis.total_entries}")
    if analysis.invalids:
        errors.append(f"Invalid card(s): {', '.join(analysis.invalids)}")
    if analysis.duplicates:
        errors.append(f"Duplicate card(s): {', '.join(analysis.duplicate_labels)}")
    if analysis.missing:
        errors.append(f"Missing card(s): {', '.join(analysis.missing)}")
    raise ValueError(". ".join(errors))


def parse_board_and_stock(input_str: str) -> tuple[list[Card], list[Card]]:
    """Parse a 52-card deck and split it into a 28-card board and 24-card stock."""
    deck = parse_deck(input_str)
    return deck[:28], deck[28:]
