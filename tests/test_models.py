import pytest

from solvers.models import (
    Card,
    DEFAULT_DECK_STR,
    STANDARD_DECK,
    analyze_deck,
    parse_board_and_stock,
    parse_deck,
)

ORDERED_DECK_STR = DEFAULT_DECK_STR


def test_card_parsing_properties_and_equality():
    ace_spades = Card("as")
    assert ace_spades.card_str == "AS"
    assert ace_spades.rank == "A"
    assert ace_spades.suit == "S"
    assert ace_spades.value == 1
    assert ace_spades.color == "BLACK"
    assert ace_spades.is_red is False
    assert ace_spades.int_id == 0
    assert ace_spades.display == "A♠️"
    assert not ace_spades.is_king()
    assert ace_spades == Card("AS")
    assert hash(ace_spades) == hash(Card("AS"))

    ten_hearts = Card(" 10h ")
    assert ten_hearts.card_str == "TH"
    assert ten_hearts.rank == "T"
    assert ten_hearts.value == 10
    assert ten_hearts.color == "RED"
    assert ten_hearts.is_red is True
    assert ten_hearts.display == "10♥️"
    assert ten_hearts == Card("TH")

    king_clubs = Card("KC")
    assert king_clubs.is_king()
    assert king_clubs.int_id == 51
    assert len({Card(c).int_id for c in STANDARD_DECK}) == 52


def test_card_adjacency_and_pyramid_pairing():
    ace = Card("AS")
    two = Card("2H")
    king = Card("KD")
    queen = Card("QC")

    assert ace.is_adjacent(two, wrap_enabled=False)
    assert ace.is_adjacent(king, wrap_enabled=True)
    assert not ace.is_adjacent(king, wrap_enabled=False)
    assert not two.is_adjacent(queen, wrap_enabled=True)

    assert ace.is_pyramid_pair(queen)
    assert Card("6S").is_pyramid_pair(Card("7D"))
    assert not king.is_pyramid_pair(ace)


@pytest.mark.parametrize("bad_token", ["", "A", "1S", "11H", "AX", "ZZ", "100S"])
def test_card_invalid_inputs_raise(bad_token):
    with pytest.raises(ValueError):
        Card(bad_token)


def test_parse_deck_and_board_stock_valid():
    deck = parse_deck(ORDERED_DECK_STR)
    assert len(deck) == 52
    assert {c.card_str for c in deck} == STANDARD_DECK

    board, stock = parse_board_and_stock(ORDERED_DECK_STR)
    assert len(board) == 28
    assert len(stock) == 24
    assert board + stock == deck


def test_parse_deck_accepts_whitespace_newlines_and_semicolons():
    mixed = ORDERED_DECK_STR.replace(", ", " ").replace("KS ", "KS\n").replace("KH ", "KH; ")
    deck = parse_deck(mixed)
    assert len(deck) == 52
    assert {c.card_str for c in deck} == STANDARD_DECK


def test_parse_deck_detects_missing_duplicates_and_invalids():
    bad_str = ORDERED_DECK_STR.replace("2S", "AS").replace("3S", "XX")
    with pytest.raises(ValueError) as exc_info:
        parse_deck(bad_str)
    msg = str(exc_info.value)
    assert "Invalid card(s): XX" in msg
    assert "Duplicate card(s): AS" in msg
    assert "2S" in msg and "3S" in msg


def test_analyze_deck_groups_duplicate_counts():
    analysis = analyze_deck("AS, AS, AS, ZZ, 10H")
    assert not analysis.is_valid
    assert analysis.total_entries == 5
    assert analysis.valid_unique == 2
    assert len(analysis.missing) == 50
    assert analysis.invalids == ("ZZ",)
    assert analysis.duplicates == (("AS", 3),)
    assert analysis.duplicate_labels == ["AS (x3)"]
