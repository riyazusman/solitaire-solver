

class Card:
    VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
              '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
    VALID_SUITS = {'S', 'H', 'D', 'C'}
    SUIT_SYMBOLS = {'S': '♠️', 'H': '♥️', 'D': '♦️', 'C': '♣️'}

    def __init__(self, card_str):
        raw_str = card_str.strip().upper()
        if raw_str.startswith('10'):
            raw_str = 'T' + raw_str[2:]
            
        self.card_str = raw_str
        
        if len(self.card_str) != 2:
            raise ValueError(f"Invalid card format '{card_str}'. Expected rank and suit (e.g., 'AH').")
            
        self.rank = self.card_str[0]
        self.suit = self.card_str[1]
        
        if self.rank not in self.VALUES:
            raise ValueError(f"Invalid rank '{self.rank}' in card '{card_str}'.")
        if self.suit not in self.VALID_SUITS:
            raise ValueError(f"Invalid suit '{self.suit}' in card '{card_str}'. Valid suits are S, H, D, C.")
            
        self.value = self.VALUES[self.rank]

    def is_adjacent(self, other_card, wrap_enabled=True):
        diff = abs(self.value - other_card.value)
        if diff == 1:
            return True
        if wrap_enabled and diff == 12:
            return True
        return False

    def is_pyramid_pair(self, other_card):
        return self.value + other_card.value == 13

    def is_king(self):
        return self.value == 13

    @property
    def display(self):
        display_rank = '10' if self.rank == 'T' else self.rank
        return f"{display_rank}{self.SUIT_SYMBOLS[self.suit]}"
    
    @property
    def color(self):
        return 'RED' if self.suit in ('H', 'D') else 'BLACK'

    def __str__(self):
        return self.card_str

    def __repr__(self):
        return f"Card('{self.card_str}')"

STANDARD_DECK = frozenset(f"{rank}{suit}" for rank in Card.VALUES for suit in Card.VALID_SUITS)

def parse_input(input_str):
    raw_cards = [c.strip() for c in input_str.split(',') if c.strip()]
    standard_deck = STANDARD_DECK
            
    parsed_cards = []
    seen = set()
    duplicates = []
    invalids = []
    
    for c in raw_cards:
        try:
            card = Card(c)
            if card.card_str in seen:
                duplicates.append(card.card_str)
            else:
                seen.add(card.card_str)
            parsed_cards.append(card)
        except ValueError:
            invalids.append(c)
            
    missing = list(standard_deck - seen)
    errors = []
    
    if len(raw_cards) != 52:
        errors.append(f"Expected exactly 52 cards, got {len(raw_cards)}")
    if invalids:
        errors.append(f"Invalid card(s): {', '.join(invalids)}")
    if duplicates:
        errors.append(f"Duplicate card(s): {', '.join(duplicates)}")
    if missing and (len(raw_cards) != 52 or duplicates or invalids):
        errors.append(f"Missing card(s): {', '.join(sorted(missing))}")
        
    if errors:
        raise ValueError(". ".join(errors))
        
    board_cards = parsed_cards[:28]
    stock_cards = parsed_cards[28:]
    
    return board_cards, stock_cards

def analyze_deck_status(input_str):
    raw_cards = [c.strip() for c in input_str.split(',') if c.strip()]
    seen = set()
    invalids = []
    duplicates = []
    for c in raw_cards:
        try:
            card = Card(c)
        except ValueError:
            invalids.append(c)
            continue
        if card.card_str in seen:
            duplicates.append(card.card_str)
        seen.add(card.card_str)
            
    missing = sorted(STANDARD_DECK - seen)
    return len(raw_cards), len(seen), missing, invalids, duplicates