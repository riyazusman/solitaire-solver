# Solitaire Solver Studio

A Streamlit app with solvers for TriPeaks, Pyramid and Freecell.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
streamlit run main_app.py
```

Pick a game from the sidebar, enter the deck, and click Solve.

## Deck input

Cards are comma-separated, e.g. `9H, 5S, 6C, ...` (`T` = 10; suits `H D C S`). Enter all 52 cards.

- **TriPeaks / Pyramid:** the first 28 cards are the board, the remaining 24 are the stock.
- **Freecell:** cards are dealt across 8 cascades, left to right, row by row.

## Options

- **Minimize Moves:** searches deeper for the shortest solution (slower).
- **Freecell goals:** clear the whole board, a specific card, or chosen ranks, each with an optional move limit.
- Searches pause after 60 seconds so you can continue or stop and keep the best result so far.

## Files

- `main_app.py` – entry point and navigation
- `tripeaks_solver.py`, `pyramid_solver.py`, `freecell_solver.py` – solvers and pages
- `shared_models.py` – `Card` and helpers
- `search_control.py` – search timing and stop/continue handling