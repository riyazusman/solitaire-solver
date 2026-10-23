from concurrent.futures import ThreadPoolExecutor
from time import monotonic

from search_control import SEARCH_PROMPT_INTERVAL_SECONDS, SearchControl, SearchStopped
from shared_models import parse_input, analyze_deck_status


class PyramidBoard:
    def __init__(self):
        # card index -> indices of the cards sitting on top of it
        self.covered_by = {i: [] for i in range(28)}
        
        overlapping_links = {
            0: [1, 2],
            1: [3, 4], 2: [4, 5],
            3: [6, 7], 4: [7, 8], 5: [8, 9],
            6: [10, 11], 7: [11, 12], 8: [12, 13], 9: [13, 14],
            10: [15, 16], 11: [16, 17], 12: [17, 18], 13: [18, 19], 14: [19, 20],
            15: [21, 22], 16: [22, 23], 17: [23, 24], 18: [24, 25], 19: [25, 26], 20: [26, 27]
        }
        
        for covered_card, covering_cards in overlapping_links.items():
            self.covered_by[covered_card].extend(covering_cards)

    def get_initial_free_cards(self):
        return [i for i, covers in self.covered_by.items() if not covers]

    def get_row_and_position(self, idx):
        if idx == 0: return 1, 1
        elif idx < 3: return 2, idx
        elif idx < 6: return 3, idx - 2
        elif idx < 10: return 4, idx - 5
        elif idx < 15: return 5, idx - 9
        elif idx < 21: return 6, idx - 14
        elif idx < 28: return 7, idx - 20
        return -1, -1


class PyramidSolver:
    def __init__(self, board_cards, stock_cards, board_layout, minimize_draws=False, max_passes=1, search_control=None):
        self.board_cards = board_cards
        self.stock_cards = stock_cards
        self.layout = board_layout
        self.minimize_draws = minimize_draws
        self.max_passes = max_passes
        self.search_control = search_control
        self.max_draws = len(self.stock_cards) * self.max_passes
        
        self.best_sequence = []
        self.max_cleared = 0
        self.min_stock_used = float('inf')
        self.visited_states = {}

    def solve(self):
        self.best_sequence = []
        self.max_cleared = 0
        self.min_stock_used = float('inf')
        
        initial_board_mask = (1 << 28) - 1
        initial_stock = tuple(range(len(self.stock_cards)))
        initial_waste = ()
        
        self.max_draws = len(self.stock_cards) * self.max_passes
        self.visited_states.clear()
        try:
            won = self._dfs(initial_board_mask, initial_stock, initial_waste, 0, 1, [], 0)
            
            if won and self.minimize_draws:
                for draw_limit in range(self.min_stock_used):
                    self.max_draws = draw_limit
                    self.visited_states.clear()
                    if self._dfs(initial_board_mask, initial_stock, initial_waste, 0, 1, [], 0):
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

    def _dfs(self, board_mask, stock_pile, waste_pile, total_draws, passes, current_path, cards_cleared):
        is_new_best = False
        if cards_cleared > self.max_cleared:
            is_new_best = True
        elif cards_cleared == self.max_cleared and total_draws < self.min_stock_used:
            is_new_best = True

        if is_new_best:
            self.max_cleared = cards_cleared
            self.min_stock_used = total_draws
            self.best_sequence = list(current_path)
            
        if cards_cleared == 28:
            return True 

        if self.search_control is not None:
            self.search_control.checkpoint()

        state_key = (board_mask, stock_pile, waste_pile, passes)
        previous_draws = self.visited_states.get(state_key)
        if previous_draws is not None and previous_draws <= total_draws:
            return False
        self.visited_states[state_key] = total_draws

        free_indices = self._get_free_board_cards(board_mask)

        # 1. Play Single King from Board
        for idx in free_indices:
            if self.board_cards[idx].is_king():
                new_mask = board_mask & ~(1 << idx)
                row, pos = self.layout.get_row_and_position(idx)
                current_path.append(f"Play {self.board_cards[idx].display} from board (Row {row}, Position {pos})")
                
                if self._dfs(new_mask, stock_pile, waste_pile, total_draws, passes, current_path, cards_cleared + 1):
                    return True
                
                current_path.pop()

        # 2. Pair Board + Board
        for i in range(len(free_indices)):
            for j in range(i + 1, len(free_indices)):
                idx1 = free_indices[i]
                idx2 = free_indices[j]
                if self.board_cards[idx1].is_pyramid_pair(self.board_cards[idx2]):
                    new_mask = board_mask & ~(1 << idx1) & ~(1 << idx2)
                    r1, p1 = self.layout.get_row_and_position(idx1)
                    r2, p2 = self.layout.get_row_and_position(idx2)
                    current_path.append(f"Pair {self.board_cards[idx1].display} (Row {r1}, Pos {p1}) and {self.board_cards[idx2].display} (Row {r2}, Pos {p2})")
                    
                    if self._dfs(new_mask, stock_pile, waste_pile, total_draws, passes, current_path, cards_cleared + 2):
                        return True
                    
                    current_path.pop()

        if waste_pile:
            top_waste_idx = waste_pile[-1]
            top_waste_card = self.stock_cards[top_waste_idx]
            
            expected_draw_str = f"Draw {top_waste_card.display} from stock"

            # 3. Play Single King from Waste
            if top_waste_card.is_king():
                new_waste = waste_pile[:-1]
                is_combined = current_path and current_path[-1] == expected_draw_str
                
                if is_combined:
                    prev_move = current_path.pop()
                    current_path.append(f"Draw and immediately play {top_waste_card.display}")
                else:
                    current_path.append(f"Play {top_waste_card.display} from waste")
                
                if self._dfs(board_mask, stock_pile, new_waste, total_draws, passes, current_path, cards_cleared):
                    return True
                
                current_path.pop()
                if is_combined:
                    current_path.append(prev_move)

            # 4. Pair Board + Waste
            for idx in free_indices:
                if self.board_cards[idx].is_pyramid_pair(top_waste_card):
                    new_mask = board_mask & ~(1 << idx)
                    new_waste = waste_pile[:-1]
                    row, pos = self.layout.get_row_and_position(idx)
                    
                    is_combined = current_path and current_path[-1] == expected_draw_str
                    
                    if is_combined:
                        prev_move = current_path.pop()
                        current_path.append(f"Draw {top_waste_card.display} and immediately pair with {self.board_cards[idx].display} from board (Row {row}, Pos {pos})")
                    else:
                        current_path.append(f"Pair {self.board_cards[idx].display} from board (Row {row}, Pos {pos}) and {top_waste_card.display} from waste")
                    
                    if self._dfs(new_mask, stock_pile, new_waste, total_draws, passes, current_path, cards_cleared + 1):
                        return True

                    current_path.pop()
                    if is_combined:
                        current_path.append(prev_move)

            # 5. Pair Waste + Waste
            if len(waste_pile) >= 2:
                top_waste_idx = waste_pile[-1]
                sub_waste_idx = waste_pile[-2]
                top_card = self.stock_cards[top_waste_idx]
                sub_card = self.stock_cards[sub_waste_idx]
                
                if top_card.is_pyramid_pair(sub_card):
                    new_waste = waste_pile[:-2]
                    
                    expected_draw_str = f"Draw {top_card.display} from stock"
                    is_combined = current_path and current_path[-1] == expected_draw_str
                    
                    if is_combined:
                        prev_move = current_path.pop()
                        current_path.append(f"Draw {top_card.display} and immediately pair with {sub_card.display} in waste")
                    else:
                        current_path.append(f"Pair {top_card.display} and {sub_card.display} in waste")
                        
                    if self._dfs(board_mask, stock_pile, new_waste, total_draws, passes, current_path, cards_cleared):
                        return True

                    current_path.pop()
                    if is_combined:
                        current_path.append(prev_move)

        # 6. Draw from Stock
        if stock_pile and total_draws < self.max_draws:
            drawn_idx = stock_pile[0]
            new_stock = stock_pile[1:]
            new_waste = waste_pile + (drawn_idx,)
            drawn_card = self.stock_cards[drawn_idx]
            current_path.append(f"Draw {drawn_card.display} from stock")
            
            if self._dfs(board_mask, new_stock, new_waste, total_draws + 1, passes, current_path, cards_cleared):
                return True
                
            current_path.pop()

        # 7. Recycle Waste to Stock
        if not stock_pile and passes < self.max_passes and waste_pile:
            # Keeps draw order: first card wasted is first drawn again
            new_stock = waste_pile
            new_waste = ()
            current_path.append(f"Recycle waste pile to stock (Pass {passes + 1}/{self.max_passes})")
            
            if self._dfs(board_mask, new_stock, new_waste, total_draws, passes + 1, current_path, cards_cleared):
                return True
                
            current_path.pop()

        return False


def _run_pyramid_search(board_cards, stock_cards, max_passes, minimize_draws, search_control):
    solver = PyramidSolver(
        board_cards=board_cards,
        stock_cards=stock_cards,
        board_layout=PyramidBoard(),
        minimize_draws=minimize_draws,
        max_passes=max_passes,
        search_control=search_control,
    )
    sequence, max_cleared = solver.solve()
    return sequence, max_cleared, solver.min_stock_used, len(solver.stock_cards)


def _display_pyramid_result(st, result):
    sequence = result["sequence"]
    max_cleared = result["max_cleared"]
    elapsed = result["elapsed"]
    stopped = result["stopped"]
    minimize_draws = result["minimize_draws"]
    max_passes = result["max_passes"]
    stock_draws = result["stock_draws"]
    stock_count = result["stock_count"]

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
        st.metric(label="Stock Draws Used", value=f"{stock_draws} / {stock_count * max_passes}")

    if max_cleared == 28 and minimize_draws and not stopped:
        st.subheader("Move Sequence (fewest stock draws):")
    elif max_cleared == 28 and stopped and minimize_draws:
        st.subheader("Winning Move Sequence (minimum draws unverified):")
    elif max_cleared == 28:
        st.subheader("Winning Move Sequence:")
    else:
        st.subheader("Best Partial Move Sequence:")

    sequence_text = ""
    for step_num, step in enumerate(sequence):
        sequence_text += f"{step_num + 1}. {step}\n"
    st.code(sequence_text, language="text")


def main():
    import streamlit as st

    st.title("🛕 Pyramid Solitaire Solver")
    st.markdown("""
    Find the optimal sequence to clear a Pyramid board by pairing cards that equal 13 (Kings are 13 and can be removed alone). 
    Enter exactly 52 comma-separated cards.
    """)

    st.markdown("**Cards 1-28:** The Board (Top row to bottom row, left to right).")
    st.code("""
           [01]             (Row 1: 1 card)
         [02][03]           (Row 2: 2 cards)
       [04][05][06]         (Row 3: 3 cards)
     [07][08][09][10]       (Row 4: 4 cards)
   [11][12][13][14][15]     (Row 5: 5 cards)
 [16][17][18][19][20][21]   (Row 6: 6 cards)
[22][23][24][25][26][27][28] (Row 7: 7 cards)
    """, language="text")

    st.markdown("**Cards 29-52:** The Stock Pile (The remaining 24 cards, left to right).")

    default_input = (
        "AS, 2S, 3S, 4S, 5S, 6S, 7S, 8S, 9S, TS, JS, QS, KS, "
        "AH, 2H, 3H, 4H, 5H, 6H, 7H, 8H, 9H, TH, JH, QH, KH, "
        "AD, 2D, 3D, 4D, 5D, 6D, 7D, 8D, 9D, TD, JD, QD, KD, "
        "AC, 2C, 3C, 4C, 5C, 6C, 7C, 8C, 9C, TC, JC, QC, KC"
    )

    card_input = st.text_area("Enter your 52 cards:", value=default_input, height=150, key="pyramid_input")

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
        max_passes = st.radio("Stock Passes (Recycles)", options=[1, 3], 
                              format_func=lambda x: "1 Pass (Strict)" if x == 1 else "3 Passes (2 Recycles)", 
                              horizontal=True)
    with col2:
        minimize_draws = st.checkbox("Minimize Stock Draws", value=False, 
                                     help="Perform a deeper search to find the absolute minimum number of stock draws required. (May take significantly longer to find the best solution). Use this for point based games.")

    job = st.session_state.get("pyramid_solver_job")
    if st.button("Solve Pyramid", type="primary", disabled=job is not None):
        try:
            board_cards, stock_cards = parse_input(card_input)
        except ValueError as e:
            st.error(f"**Input Error:** {e}")
        except Exception as e:
            st.error("**An unexpected error occurred.**")
            st.exception(e)
        else:
            st.session_state.pop("pyramid_solver_result", None)
            search_control = SearchControl()
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _run_pyramid_search,
                board_cards,
                stock_cards,
                max_passes,
                minimize_draws,
                search_control,
            )
            started_at = monotonic()
            st.session_state.pyramid_solver_job = {
                "future": future,
                "executor": executor,
                "search_control": search_control,
                "started_at": started_at,
                "interval_started_at": started_at,
                "resume_pending": False,
                "stop_requested": False,
                "stop_requested_at": None,
                "finished_at": None,
                "max_passes": max_passes,
                "minimize_draws": minimize_draws,
            }

    job = st.session_state.get("pyramid_solver_job")
    if job is not None:
        @st.fragment(run_every="1s")
        def render_search_status():
            if job.get("result_rendered"):
                result = st.session_state.get("pyramid_solver_result")
                if result is not None:
                    _display_pyramid_result(st, result)
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
                    st.session_state.pop("pyramid_solver_job", None)
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
                    "max_passes": job["max_passes"],
                    "stock_draws": stock_draws,
                    "stock_count": stock_count,
                }
                st.session_state.pyramid_solver_result = result
                st.session_state.pop("pyramid_solver_job", None)
                job["result_rendered"] = True
                _display_pyramid_result(st, result)
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
                    if st.button("Continue searching", key="pyramid_continue_search"):
                        job["interval_started_at"] = monotonic()
                        job["resume_pending"] = True
                        search_control.resume_requested.set()
                with stop_col:
                    if st.button("Stop and show best result", key="pyramid_stop_search"):
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
                        if st.button("Continue searching", key="pyramid_continue_search"):
                            job["interval_started_at"] = monotonic()
                            job["resume_pending"] = True
                            search_control.resume_requested.set()
                    with stop_col:
                        if st.button("Stop and show best result", key="pyramid_stop_search"):
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
    elif "pyramid_solver_result" in st.session_state:
        _display_pyramid_result(st, st.session_state.pyramid_solver_result)


if __name__ == "__main__":
    main()