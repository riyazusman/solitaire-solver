from concurrent.futures import ThreadPoolExecutor
from time import monotonic

from search_control import SEARCH_PROMPT_INTERVAL_SECONDS, SearchControl, SearchStopped
from shared_models import parse_input, analyze_deck_status

class TriPeaksBoard:
    def __init__(self):
        self.covered_by = {i: [] for i in range(28)}
        overlapping_links = {
            0: [3, 4], 1: [5, 6], 2: [7, 8],
            3: [9, 10], 4: [10, 11], 5: [12, 13], 6: [13, 14], 7: [15, 16], 8: [16, 17],
            9: [18, 19], 10: [19, 20], 11: [20, 21], 
            12: [21, 22], 13: [22, 23], 14: [23, 24], 
            15: [24, 25], 16: [25, 26], 17: [26, 27]
        }
        for covered_card, covering_cards in overlapping_links.items():
            self.covered_by[covered_card].extend(covering_cards)

    def get_initial_free_cards(self):
        return [i for i, covers in self.covered_by.items() if not covers]

    def get_row_and_position(self, idx):
        if idx < 3: return 1, idx + 1
        elif idx < 9: return 2, idx - 2
        elif idx < 18: return 3, idx - 8
        elif idx < 28: return 4, idx - 17
        return -1, -1

class TriPeaksSolver:
    def __init__(self, board_cards, stock_cards, board_layout, wrap_enabled=True, minimize_draws=False, search_control=None):
        self.board_cards = board_cards
        self.initial_waste = stock_cards[0]
        self.stock_cards = stock_cards[1:]
        self.layout = board_layout
        self.wrap_enabled = wrap_enabled
        self.minimize_draws = minimize_draws
        self.search_control = search_control
        self.max_draws = len(self.stock_cards)
        self.best_sequence = []
        self.max_cleared = 0
        self.min_stock_used = float('inf')
        self.visited_states = set()

    def solve(self):
        self.best_sequence = []
        self.max_cleared = 0
        self.min_stock_used = float('inf')
        
        initial_board_mask = (1 << 28) - 1
        initial_path = [f"Start: Waste pile is {self.initial_waste.display}"]
        
        self.max_draws = len(self.stock_cards)
        self.visited_states.clear()
        try:
            won = self._dfs(initial_board_mask, self.initial_waste, 0, list(initial_path), 0)
            
            if won and self.minimize_draws:
                for draw_limit in range(self.min_stock_used):
                    self.max_draws = draw_limit
                    self.visited_states.clear()
                    if self._dfs(initial_board_mask, self.initial_waste, 0, list(initial_path), 0):
                        break
        except SearchStopped:
            pass
        
        return self.best_sequence, self.max_cleared

    def _get_free_board_cards(self, board_mask):
        free_indices = []
        for i in range(28):
            if board_mask & (1 << i):
                is_covered = False
                for covering_card in self.layout.covered_by[i]:
                    if board_mask & (1 << covering_card):
                        is_covered = True
                        break
                if not is_covered:
                    free_indices.append(i)
        return free_indices

    def _dfs(self, board_mask, waste_card, stock_drawn, current_path, cards_cleared):
        is_new_best = False
        if cards_cleared > self.max_cleared:
            is_new_best = True
        elif cards_cleared == self.max_cleared and stock_drawn < self.min_stock_used:
            is_new_best = True

        if is_new_best:
            self.max_cleared = cards_cleared
            self.min_stock_used = stock_drawn
            self.best_sequence = list(current_path)
            
        if cards_cleared == 28:
            return True 

        if self.search_control is not None:
            self.search_control.checkpoint()

        state_key = (board_mask, waste_card.value, stock_drawn)
        if state_key in self.visited_states:
            return False
        self.visited_states.add(state_key)

        free_indices = self._get_free_board_cards(board_mask)
        valid_board_moves = [idx for idx in free_indices if waste_card.is_adjacent(self.board_cards[idx], self.wrap_enabled)]
        
        valid_board_moves.sort(key=lambda idx: self._count_unblocked_cards(board_mask, idx), reverse=True)

        for idx in valid_board_moves:
            candidate_card = self.board_cards[idx]
            new_mask = board_mask & ~(1 << idx)
            row, pos = self.layout.get_row_and_position(idx)
            current_path.append(f"Play {candidate_card.display} from board (Row {row}, Position {pos})")
            
            if self._dfs(new_mask, candidate_card, stock_drawn, current_path, cards_cleared + 1):
                return True
            
            current_path.pop()

        if stock_drawn < self.max_draws:
            new_waste = self.stock_cards[stock_drawn]
            current_path.append(f"Draw {new_waste.display} from stock")
            
            if self._dfs(board_mask, new_waste, stock_drawn + 1, current_path, cards_cleared):
                return True
                
            current_path.pop()

        return False

    def _count_unblocked_cards(self, board_mask, played_idx):
        unblocked_count = 0
        for i in range(28):
            if not (board_mask & (1 << i)):
                continue 
            active_parents = [p for p in self.layout.covered_by[i] if (board_mask & (1 << p))]
            if len(active_parents) == 1 and active_parents[0] == played_idx:
                unblocked_count += 1
        return unblocked_count


def _run_tripeaks_search(board_cards, stock_cards, wrap_enabled, minimize_draws, search_control):
    solver = TriPeaksSolver(
        board_cards=board_cards,
        stock_cards=stock_cards,
        board_layout=TriPeaksBoard(),
        wrap_enabled=wrap_enabled,
        minimize_draws=minimize_draws,
        search_control=search_control,
    )
    sequence, max_cleared = solver.solve()
    return sequence, max_cleared, solver.min_stock_used, len(solver.stock_cards)


def _display_tripeaks_result(st, result):
    max_cleared = result["max_cleared"]
    stopped = result["stopped"]
    minimize_draws = result["minimize_draws"]
    elapsed = result["elapsed"]

    st.divider()
    if max_cleared == 28:
        if stopped and minimize_draws:
            st.warning(f"🎉 Perfect clear found in {elapsed:.2f}s, but the search stopped before minimum stock draws were proven.")
        else:
            st.success(f"🎉 Perfect Clear Found! (Calculated in {elapsed:.2f}s)")
    elif stopped:
        st.warning(f"Search stopped after {elapsed:.2f}s. Best result found so far; optimality was not proven.")
    else:
        st.warning(f"⚠️ No perfect clear exists. Best partial clear found in {elapsed:.2f}s.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Cards Cleared from Board", value=f"{max_cleared} / 28")
    with col2:
        st.metric(label="Stock Draws Used", value=f"{result['stock_draws']} / {result['stock_count']}")

    if max_cleared == 28 and minimize_draws and not stopped:
        st.subheader("Move Sequence (fewest stock draws):")
    elif max_cleared == 28 and stopped and minimize_draws:
        st.subheader("Winning Move Sequence (minimum draws unverified):")
    elif max_cleared == 28:
        st.subheader("Winning Move Sequence:")
    else:
        st.subheader("Best Partial Move Sequence:")

    sequence_text = ""
    for step_num, step in enumerate(result["sequence"]):
        sequence_text += f"{step_num}. {step}\n"
    st.code(sequence_text, language="text")


def main():
    import streamlit as st

    st.title("🏔️ Tripeaks Solitaire Solver")
    st.markdown("""
    Find the optimal sequence to clear a Tripeaks board. 
    Enter exactly 52 comma-separated cards (e.g., `AS, 2H, 10C, KD...`).
    """)

    st.markdown("**Cards 1-28:** The Board (Top row to bottom row, left to right).")
    st.code("""
      [01]         [02]         [03]          (Row 1: 3 cards)
    [04][05]     [06][07]     [08][09]        (Row 2: 6 cards)
  [10][11][12] [13][14][15] [16][17][18]      (Row 3: 9 cards)
 [19][20][21][22][23][24][25][26][27][28]     (Row 4: 10 cards)
    """, language="text")

    st.markdown("**Cards 29-52:** The Stock Pile (The remaining 24 cards, left to right).")

    default_input = (
        "AS, 2S, 3S, 4S, 5S, 6S, 7S, 8S, 9S, TS, JS, QS, KS, "
        "AH, 2H, 3H, 4H, 5H, 6H, 7H, 8H, 9H, TH, JH, QH, KH, "
        "AD, 2D, 3D, 4D, 5D, 6D, 7D, 8D, 9D, TD, JD, QD, KD, "
        "AC, 2C, 3C, 4C, 5C, 6C, 7C, 8C, 9C, TC, JC, QC, KC"
    )

    card_input = st.text_area("Enter your 52 cards:", value=default_input, height=150, key="tripeaks_input")

    total_entered, valid_unique, missing_cards, invalid_cards, duplicate_cards = analyze_deck_status(card_input)

    if total_entered == 52 and valid_unique == 52:
        st.success("✅ **Deck Status:** 52 valid cards ready to go!")
    else:
        st.info(f"📊 **Deck Status:** {total_entered}/52 entries | {valid_unique} valid unique cards")
        for label, cards in (
            ("Invalid entries", invalid_cards),
            ("Duplicate cards", duplicate_cards),
            ("Missing cards", missing_cards),
        ):
            if cards:
                st.warning(f"**{label} ({len(cards)}):** {', '.join(cards)}")

    col1, col2 = st.columns(2)
    with col1:
        wrap_enabled = st.checkbox("Enable King-Ace Wrapping", value=True, help="Allow playing an Ace on a King and vice versa.")
    with col2:
        minimize_draws = st.checkbox("Minimize Stock Draws", value=False, help="Perform a deeper search to find the absolute minimum number of stock draws required. (May take significantly longer to find the best solution). Use this for point based games.")

    job = st.session_state.get("tripeaks_solver_job")
    if st.button("Solve Tripeaks", type="primary", disabled=job is not None):
        try:
            board_cards, stock_cards = parse_input(card_input)
        except ValueError as e:
            st.error(f"**Input Error:** {e}")
        except Exception as e:
            st.error("**An unexpected error occurred.**")
            st.exception(e)
        else:
            st.session_state.pop("tripeaks_solver_result", None)
            search_control = SearchControl()
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _run_tripeaks_search,
                board_cards,
                stock_cards,
                wrap_enabled,
                minimize_draws,
                search_control,
            )
            started_at = monotonic()
            st.session_state.tripeaks_solver_job = {
                "future": future,
                "executor": executor,
                "search_control": search_control,
                "started_at": started_at,
                "interval_started_at": started_at,
                "resume_pending": False,
                "stop_requested": False,
                "stop_requested_at": None,
                "finished_at": None,
                "minimize_draws": minimize_draws,
            }

    job = st.session_state.get("tripeaks_solver_job")
    if job is not None:
        @st.fragment(run_every="1s")
        def render_search_status():
            if job.get("result_rendered"):
                result = st.session_state.get("tripeaks_solver_result")
                if result is not None:
                    _display_tripeaks_result(st, result)
                return

            future = job["future"]
            search_control = job["search_control"]

            if future.done():
                job["executor"].shutdown(wait=False)
                if job["finished_at"] is None:
                    job["finished_at"] = monotonic()
                try:
                    sequence, max_cleared, stock_draws, stock_count = future.result()
                except Exception as e:
                    st.session_state.pop("tripeaks_solver_job", None)
                    job["result_rendered"] = True
                    st.error("**An unexpected error occurred.**")
                    st.exception(e)
                    return

                elapsed_until = job["stop_requested_at"] or job["finished_at"]
                result = {
                    "sequence": sequence,
                    "max_cleared": max_cleared,
                    "elapsed": elapsed_until - job["started_at"],
                    "stopped": job["stop_requested"],
                    "minimize_draws": job["minimize_draws"],
                    "stock_draws": stock_draws,
                    "stock_count": stock_count,
                }
                st.session_state.tripeaks_solver_result = result
                st.session_state.pop("tripeaks_solver_job", None)
                job["result_rendered"] = True
                _display_tripeaks_result(st, result)
                return

            if job["stop_requested"]:
                st.info("Stopping search and collecting the best result found so far...")
                return

            if job["resume_pending"]:
                if search_control.paused.is_set():
                    st.status("Resuming the search...", state="running", expanded=False)
                    return
                job["resume_pending"] = False

            if search_control.paused.is_set():
                st.info("Search paused. Continue searching or stop and show the best result so far.")
                continue_col, stop_col = st.columns(2)
                with continue_col:
                    if st.button("Continue searching", key="tripeaks_continue_search"):
                        job["interval_started_at"] = monotonic()
                        job["resume_pending"] = True
                        search_control.resume_requested.set()
                with stop_col:
                    if st.button("Stop and show best result", key="tripeaks_stop_search"):
                        job["stop_requested"] = True
                        job["stop_requested_at"] = monotonic()
                        search_control.stop_requested.set()
                        search_control.resume_requested.set()
                return

            if search_control.pause_requested.is_set():
                if search_control.paused.is_set():
                    st.info("Search paused. Continue searching or stop and show the best result so far.")
                    continue_col, stop_col = st.columns(2)
                    with continue_col:
                        if st.button("Continue searching", key="tripeaks_continue_search"):
                            job["interval_started_at"] = monotonic()
                            job["resume_pending"] = True
                            search_control.resume_requested.set()
                    with stop_col:
                        if st.button("Stop and show best result", key="tripeaks_stop_search"):
                            job["stop_requested"] = True
                            job["stop_requested_at"] = monotonic()
                            search_control.stop_requested.set()
                            search_control.resume_requested.set()
                else:
                    elapsed = int(monotonic() - job["started_at"])
                    st.status(
                        f"Pausing at the next search checkpoint... {elapsed}s elapsed",
                        state="running",
                        expanded=False,
                    )
                return

            elapsed = monotonic() - job["interval_started_at"]
            if elapsed >= SEARCH_PROMPT_INTERVAL_SECONDS:
                search_control.pause_requested.set()
                elapsed_total = int(monotonic() - job["started_at"])
                st.status(
                    f"Pausing at the next search checkpoint... {elapsed_total}s elapsed",
                    state="running",
                    expanded=False,
                )
            else:
                elapsed_total = int(monotonic() - job["started_at"])
                st.status(
                    f"Searching for a solution... {elapsed_total}s elapsed",
                    state="running",
                    expanded=False,
                )

        render_search_status()
    elif "tripeaks_solver_result" in st.session_state:
        _display_tripeaks_result(st, st.session_state.tripeaks_solver_result)


if __name__ == "__main__":
    main()