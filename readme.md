# Solitaire Solver Studio

A Streamlit app with solvers, board previews, and interactive step-by-step playback for **TriPeaks**, **Pyramid**, and **Freecell** Solitaire.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run main_app.py
```

Pick a game from the sidebar—or link directly to `/tripeaks`, `/pyramid`, or `/freecell`—enter (or generate) a 52-card deck, preview the layout, and click **Solve**.

## Deck Input

Enter all 52 cards separated by **commas, spaces, semicolons, or newlines**, e.g. `9H, 5S, 10C, KD ...`.
Both `10` and `T` are accepted for tens; suits are `S H D C` (case-insensitive).

- **TriPeaks:** Cards 1–28 form the 4-row peaks (top to bottom, left to right). **Card 29** starts face-up on the waste pile, and Cards 30–52 form the 23-card stock pile.
- **Pyramid:** Cards 1–28 form the 7-row pyramid (top to bottom, left to right). Cards 29–52 form the 24-card stock pile.
- **Freecell:** Cards 1–52 are dealt across 8 cascades, left to right, row by row.

Quick-fill buttons (**Random Deal**, **Sample Deal**, **Ordered Deck**, **Clear**) and an expandable **Board Preview** are available on every page.

## Game Options & Search Controls

- **TriPeaks:** King–Ace Wrapping toggle; **Minimize Stock Draws** (Layered 0-1 BFS).
- **Pyramid:** Stock Passes (`1`, `2`, `3`, or `Unlimited`), **Minimize Stock Draws** (binary search), and **Standard Waste Rules** toggle (strict drawn-card + waste-top pairing vs. relaxed waste pairing).
- **Freecell:** Goals (clear whole board, single target card, or chosen ranks), optional **Move Limit**, **Minimize Moves** (weighted best-first → A\*), **Allow Foundation → Cascade**, and **Atomic Move Counting**.
- **Search Controls:** Active search timer (excluding paused time), mid-search **Stop**, configurable pause prompt interval (`30s`–`300s`), and configurable memory safety state cap. Solutions can be copied as text or stepped through visually in the **Step-by-Step Playback** tab.

## Docker

Build and run with the default `/solitaire-solver` base path:

```bash
docker build -t solitaire-solver .
docker run -p 8501:8501 solitaire-solver
# Open http://localhost:8501/solitaire-solver
```

To serve at root (`http://localhost:8501/`) instead:

```bash
docker run -p 8501:8501 -e STREAMLIT_SERVER_BASE_URL_PATH="" solitaire-solver
```

## Testing & Benchmarks

```bash
pip install -r requirements-dev.txt
pytest -v
python tests/benchmarks.py
```

## Project Layout

- `main_app.py` – Streamlit entry point and multipage navigation
- `solvers/` – pure-Python solver engines (`models.py`, `search_control.py`, `tripeaks.py`, `pyramid.py`, `freecell.py`, `replay.py`)
- `ui/` – shared Streamlit components (`deck_input.py`, `job_runner.py`, `board_views.py`) and game pages (`ui/pages/`)
- `tests/` – unit tests, independent move-legality verifiers (`verifiers.py`), and benchmarks
