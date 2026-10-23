import heapq
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

from search_control import SEARCH_PROMPT_INTERVAL_SECONDS, SearchControl, SearchStopped
from shared_models import Card, parse_input, analyze_deck_status

class FreecellState:
    def __init__(self, deck_cards=None):
        self.cascades = [[] for _ in range(8)]
        self.freecells = []
        # highest rank placed per suit (0-13)
        self.foundations = {'S': 0, 'H': 0, 'D': 0, 'C': 0}
        
        if deck_cards:
            if len(deck_cards) != 52:
                raise ValueError("Freecell requires exactly 52 cards.")
            for i, card in enumerate(deck_cards):
                self.cascades[i % 8].append(card)

    def is_solved(self):
        return all(rank == 13 for rank in self.foundations.values())

    def get_state_key(self):
        cascade_tuples = []
        for cascade in self.cascades:
            cascade_tuples.append(tuple(c.card_str for c in cascade))
            
        frozen_cascades = tuple(sorted(cascade_tuples))
        frozen_freecells = tuple(sorted(c.card_str for c in self.freecells))
        frozen_foundations = tuple(sorted(self.foundations.items()))
        
        return (frozen_cascades, frozen_freecells, frozen_foundations)


class FreecellSolver:
    def __init__(self, initial_state, search_control=None, minimize_moves=False, target_card=None, move_limit=None,
                 target_ranks=None, rank_count=None):
        """
        target_card: get only that card to its foundation (auto-play off, each action is one move).
        target_ranks/rank_count: get rank_count (default 4) cards of each of those ranks (values 1-13) up.
        move_limit: maximum number of moves, for any goal.
        """
        self.state = initial_state
        self.search_control = search_control
        self.minimize_moves = minimize_moves
        self.target_card = target_card
        self.move_limit = move_limit
        self.target_ranks = sorted(set(target_ranks)) if target_ranks else None
        self.rank_count = rank_count or 4
        self.whole_board = target_card is None and not self.target_ranks
        self.visited_states = {}
        self.winning_sequence = []

    def _goal_remaining(self, foundations=None):
        """Lower bound on moves still needed (0 only when the goal is met)."""
        found = self.state.foundations if foundations is None else foundations
        if self.whole_board:
            return 52 - sum(found.values())
        if self.target_card is not None:
            return max(0, self.target_card.value - found[self.target_card.suit])
        have = [sum(1 for s in found if found[s] >= r) for r in self.target_ranks]
        if all(h >= self.rank_count for h in have):
            return 0
        # Relaxation: only the total number of missing cards matters; suits are combined with a small DP
        need = sum(self.rank_count - min(h, self.rank_count) for h in have)
        inf = 10 ** 6
        best = [0] + [inf] * need
        for suit, height in found.items():
            wanted = [r for r in self.target_ranks if r > height]
            options = [(0, 0)] + [(k, r - height) for k, r in enumerate(wanted, 1)]
            nxt = [inf] * (need + 1)
            for got in range(need + 1):
                if best[got] >= inf:
                    continue
                for k, cost in options:
                    reached = min(need, got + k)
                    nxt[reached] = min(nxt[reached], best[got] + cost)
            best = nxt
        return max(1, best[need])

    def _is_goal(self):
        return self._goal_remaining() == 0

    def solve(self):
        """
        Iterative best-first search. With minimize_moves, later passes weight path length more and
        prune states that cannot beat the best solution; the last pass is A*.
        proven is True when the search space was exhausted.
        Returns (path, solved, proven).
        """
        self.winning_sequence = []
        self.best_solution = None
        self.best_partial = []
        self.best_found = -1
        self._initial_node = self._snapshot()
        self._reexpand = self.minimize_moves or self.move_limit is not None
        proven = False
        passes = [(0.5, False)]
        if self.minimize_moves:
            passes += [(2.0, False), (6.0, False), (1.0, True)]

        try:
            for weight, final in passes:
                if self.best_solution is not None:
                    bound = len(self.best_solution)
                elif self.move_limit is not None:
                    bound = self.move_limit + 1
                else:
                    bound = float('inf')
                exhausted = self._search_pass(weight, final, bound)
                if self.best_solution is not None and not self.minimize_moves:
                    break
                if exhausted:
                    proven = True
                    break
        except SearchStopped:
            pass

        solved = self.best_solution is not None
        path = self.best_solution if solved else self.best_partial
        self.winning_sequence = path
        return path, solved, proven

    def _search_pass(self, weight, final, best_len):
        """Returns True if the queue was exhausted."""
        self.visited_states.clear()

        # parents[i] = (parent_index, move_texts, moves_so_far)
        parents = [(-1, [], 0)]
        counter = 0
        self._load(self._initial_node)
        start_texts = self._auto_foundation() if self.whole_board else []
        parents.append((0, start_texts, len(start_texts)))
        heap = [(self._priority(len(start_texts), weight, final), counter, 1, self._snapshot())]
        self.visited_states[self.state.get_state_key()] = len(start_texts)

        def path_to(index):
            chunks = []
            while index > 0:
                chunks.append(parents[index][1])
                index = parents[index][0]
            return [text for chunk in reversed(chunks) for text in chunk]

        best_partial_index = None
        if self._is_goal():
            self.best_solution = path_to(1)
            return True

        try:
            while heap:
                if self.search_control is not None:
                    self.search_control.checkpoint()

                priority, _, index, node = heapq.heappop(heap)
                depth = parents[index][2]
                if final and priority >= best_len:
                    return True
                if depth + self._remaining(node) >= best_len:
                    continue

                self._load(node)
                for move in self._generate_valid_moves():
                    self._apply_move(move)
                    texts = [move[5]]
                    if self.whole_board:
                        texts += self._auto_foundation()
                    child_depth = depth + len(texts)
                    key = self.state.get_state_key()
                    known = self.visited_states.get(key)
                    remaining = self._goal_remaining()
                    if (known is None or (self._reexpand and child_depth < known)) and child_depth + remaining < best_len:
                        self.visited_states[key] = child_depth
                        parents.append((index, texts, child_depth))
                        child_index = len(parents) - 1
                        if -remaining > self.best_found:
                            self.best_found = -remaining
                            best_partial_index = child_index
                        if remaining == 0:
                            best_len = child_depth
                            self.best_solution = path_to(child_index)
                            if not final:
                                return False
                        else:
                            counter += 1
                            heapq.heappush(
                                heap,
                                (self._priority(child_depth, weight, final), counter, child_index, self._snapshot()),
                            )
                    self._load(node)
            return True
        finally:
            if best_partial_index is not None:
                self.best_partial = path_to(best_partial_index)

    def _remaining(self, node):
        return self._goal_remaining(dict(node[2]))

    def _auto_foundation(self):
        """Moves safe cards to the foundations; returns the move texts."""
        texts = []
        state = self.state
        found = state.foundations
        progress = True
        while progress:
            progress = False
            candidates = [('cascade', i, c[-1]) for i, c in enumerate(state.cascades) if c]
            candidates += [('freecell', i, c) for i, c in enumerate(state.freecells)]
            for src_type, src_idx, card in candidates:
                if found[card.suit] != card.value - 1:
                    continue
                opposite = ('H', 'D') if card.suit in ('S', 'C') else ('S', 'C')
                if card.value > 2 and any(found[s] < card.value - 1 for s in opposite):
                    continue
                if src_type == 'cascade':
                    state.cascades[src_idx].pop()
                    where = f"Cascade {src_idx + 1}"
                else:
                    state.freecells.pop(src_idx)
                    where = "Freecell"
                found[card.suit] += 1
                texts.append(f"Move {card.display} from {where} to Foundation (auto)")
                progress = True
                break
        return texts

    def _snapshot(self):
        return (
            tuple(tuple(c) for c in self.state.cascades),
            tuple(self.state.freecells),
            tuple(self.state.foundations.items()),
        )

    def _load(self, node):
        cascades, freecells, foundations = node
        self.state.cascades = [list(c) for c in cascades]
        self.state.freecells = list(freecells)
        self.state.foundations = dict(foundations)

    def _priority(self, depth, weight=0.5, final=False):
        """Lower is better. The final pass is A* (depth + cards left)."""
        state = self.state
        remaining = self._goal_remaining()
        if final:
            return depth + remaining
        buried = 0
        disorder = 0
        target = self.target_card
        top_rank = self.target_ranks[-1] if self.target_ranks else None
        for cascade in state.cascades:
            lowest_below = 99
            for pos, card in enumerate(cascade):
                if card.value == state.foundations[card.suit] + 1 and (
                    self.whole_board
                    or (target is not None and card.suit == target.suit and card.value <= target.value)
                    or (top_rank is not None and card.value <= top_rank)
                ):
                    buried += len(cascade) - pos - 1
                if card.value > lowest_below:
                    disorder += 1
                lowest_below = min(lowest_below, card.value)
        if not self.whole_board:
            disorder = 0
        return 10 * remaining + 3 * buried + disorder + 2 * len(state.freecells) + depth * weight

    def _generate_valid_moves(self):
        moves = []
        state = self.state
        free_cells = 4 - len(state.freecells)
        empty_columns = sum(1 for c in state.cascades if not c)

        for i, cascade in enumerate(state.cascades):
            if not cascade:
                continue
            card = cascade[-1]

            if state.foundations[card.suit] == card.value - 1:
                moves.append(('cascade', i, 'foundation', card.suit, (card,), f"Move {card.display} from Cascade {i+1} to Foundation"))

            # length of the alternating-color descending run at the bottom
            run = 1
            while (
                run < len(cascade)
                and cascade[-run - 1].color != cascade[-run].color
                and cascade[-run - 1].value == cascade[-run].value + 1
            ):
                run += 1

            empty_used = False
            for j, target in enumerate(state.cascades):
                if i == j:
                    continue
                if not target:
                    # empty columns are interchangeable
                    if empty_used:
                        continue
                    empty_used = True
                    capacity = (free_cells + 1) * 2 ** (empty_columns - 1)
                    for n in range(1, min(run, capacity, len(cascade) - 1) + 1):
                        moves.append(self._cascade_move(i, j, n, None))
                else:
                    capacity = (free_cells + 1) * 2 ** empty_columns
                    target_card = target[-1]
                    n = target_card.value - card.value
                    if 1 <= n <= min(run, capacity):
                        bottom = cascade[-n]
                        if bottom.color != target_card.color and bottom.value == target_card.value - 1:
                            moves.append(self._cascade_move(i, j, n, target_card))

            if free_cells > 0:
                moves.append(('cascade', i, 'freecell', -1, (card,), f"Move {card.display} from Cascade {i+1} to Freecell"))

        for i, card in enumerate(state.freecells):
            if state.foundations[card.suit] == card.value - 1:
                moves.append(('freecell', i, 'foundation', card.suit, (card,), f"Move {card.display} from Freecell to Foundation"))

            empty_used = False
            for j, target in enumerate(state.cascades):
                if not target:
                    if empty_used:
                        continue
                    empty_used = True
                    moves.append(('freecell', i, 'cascade', j, (card,), f"Move {card.display} from Freecell to empty Cascade {j+1}"))
                else:
                    target_card = target[-1]
                    if target_card.color != card.color and target_card.value == card.value + 1:
                        moves.append(('freecell', i, 'cascade', j, (card,), f"Move {card.display} from Freecell to Cascade {j+1} on {target_card.display}"))

        return moves

    def _cascade_move(self, src, dest, count, target_card):
        cards = tuple(self.state.cascades[src][-count:])
        if count == 1:
            what = cards[0].display
        else:
            what = f"{count} cards ({cards[0].display} to {cards[-1].display})"
        where = f"empty Cascade {dest+1}" if target_card is None else f"Cascade {dest+1} on {target_card.display}"
        return ('cascade', src, 'cascade', dest, cards, f"Move {what} from Cascade {src+1} to {where}")

    def _apply_move(self, move):
        src_type, src_idx, dest_type, dest_idx, cards, _ = move

        if src_type == 'cascade':
            del self.state.cascades[src_idx][-len(cards):]
        elif src_type == 'freecell':
            self.state.freecells.pop(src_idx)

        if dest_type == 'foundation':
            self.state.foundations[dest_idx] += 1
        elif dest_type == 'cascade':
            self.state.cascades[dest_idx].extend(cards)
        elif dest_type == 'freecell':
            self.state.freecells.append(cards[0])


def _run_freecell_search(deck_cards, minimize_moves, search_control, target_card=None, move_limit=None,
                         target_ranks=None, rank_count=None):
    initial_state = FreecellState(deck_cards)
    solver = FreecellSolver(
        initial_state=initial_state,
        search_control=search_control,
        minimize_moves=minimize_moves,
        target_card=target_card,
        move_limit=move_limit,
        target_ranks=target_ranks,
        rank_count=rank_count,
    )
    return solver.solve()


def _display_freecell_result(st, result):
    sequence = result["sequence"]
    is_solved = result["is_solved"]
    elapsed = result["elapsed"]
    stopped = result["stopped"]
    minimize_moves = result["minimize_moves"]
    goal = result.get("goal_text")
    move_limit = result.get("move_limit")
    limit_text = f" within {move_limit} moves" if move_limit else ""

    st.divider()
    if goal is not None:
        if is_solved and result["proven_optimal"] and minimize_moves:
            st.success(f"🎯 Fewest-move path to get {goal} proven! (Calculated in {elapsed:.2f}s)")
        elif is_solved and minimize_moves:
            st.success(f"🎯 Path to {goal} found after {elapsed:.2f}s. Search stopped, so it may not be the fewest moves possible.")
        elif is_solved:
            st.success(f"🎯 Path to {goal} found in {elapsed:.2f}s. Tick 'Minimize Moves' to look for a shorter one.")
        elif stopped:
            st.warning(f"Search stopped after {elapsed:.2f}s without reaching {goal}. Closest progress found so far:")
        else:
            st.warning(f"⚠️ {goal} cannot be reached{limit_text} with the supported moves ({elapsed:.2f}s). Closest progress found:")
    elif not is_solved and not stopped and move_limit:
        st.warning(
            f"⚠️ The board cannot be cleared{limit_text} with the supported moves ({elapsed:.2f}s). "
            f"Showing the furthest progress reached:"
        )
    elif is_solved and minimize_moves and result["proven_optimal"]:
        st.success(f"🎉 Fewest-move solution proven! (Calculated in {elapsed:.2f}s)")
    elif is_solved and minimize_moves:
        st.success(f"🎉 Solution found after {elapsed:.2f}s. Search stopped, so it may not be the fewest moves possible.")
    elif is_solved:
        st.success(f"🎉 Perfect Clear Found! (Calculated in {elapsed:.2f}s)")
    elif stopped:
        st.warning(f"Search stopped after {elapsed:.2f}s. Best partial sequence found so far:")
    else:
        st.warning(
            f"⚠️ No solution found. The search explored every position reachable with the supported moves "
            f"({elapsed:.2f}s). Showing the furthest progress reached:"
        )

    st.metric(label="Total Moves", value=len(sequence))

    if sequence:
        st.subheader("Move Sequence:")
        sequence_text = ""
        for step_num, step in enumerate(sequence):
            sequence_text += f"{step_num + 1}. {step}\n"
        st.code(sequence_text, language="text")
    else:
        st.info("No valid moves could be made from the starting layout.")


def main():
    import streamlit as st

    st.title("🃏 Freecell Solitaire Solver")
    st.markdown("""
    Find the optimal sequence to clear a Freecell board. 
    Enter exactly 52 comma-separated cards. The first 8 cards form the top row of the cascades, the next 8 form the second row, and so on.
    """)

    st.markdown("**Freecell Deal Structure:**")
    st.code("""
    [C1] [C2] [C3] [C4] [C5] [C6] [C7] [C8]
     01   02   03   04   05   06   07   08
     09   10   11   12   13   14   15   16
     ...
     49   50   51   52
    """, language="text")

    default_input = (
        "AS, 2S, 3S, 4S, 5S, 6S, 7S, 8S, "
        "9S, TS, JS, QS, KS, AH, 2H, 3H, "
        "4H, 5H, 6H, 7H, 8H, 9H, TH, JH, "
        "QH, KH, AD, 2D, 3D, 4D, 5D, 6D, "
        "7D, 8D, 9D, TD, JD, QD, KD, AC, "
        "2C, 3C, 4C, 5C, 6C, 7C, 8C, 9C, "
        "TC, JC, QC, KC"
    )

    card_input = st.text_area("Enter your 52 cards (left to right, top to bottom):", value=default_input, height=150, key="freecell_input")

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

    goal_mode = st.radio(
        "Goal",
        ["Clear the whole board", "Get one card to the foundation", "Get cards of chosen ranks to the foundation"],
        key="freecell_goal_mode",
        horizontal=True,
    )
    target_text, ranks_text, move_limit, rank_count = "", "", None, None
    one_card = goal_mode == "Get one card to the foundation"
    by_ranks = goal_mode == "Get cards of chosen ranks to the foundation"
    if one_card:
        target_text = st.text_input("Target card (e.g. 4D, TC)", key="freecell_target_card")
    elif by_ranks:
        ranks_col, count_col = st.columns(2)
        with ranks_col:
            ranks_text = st.text_input("Ranks (e.g. 6 or 6, 8, K)", key="freecell_target_ranks")
        with count_col:
            rank_count = int(st.number_input(
                "How many of each rank",
                min_value=1, max_value=4, value=1, step=1, key="freecell_rank_count",
            ))
    limit_value = st.number_input(
        "Move limit (0 = no limit)", min_value=0, max_value=500, value=0, step=1, key="freecell_move_limit"
    )
    move_limit = int(limit_value) or None
    if one_card or by_ranks:
        st.caption("Auto-play is off in this mode. Every action counts as one move, including multi-card moves and foundation moves.")
    else:
        st.caption("Moves automatically sent to the foundation are counted as moves when a move limit is set.")

    minimize_moves = st.checkbox(
        "Minimize Moves",
        value=False,
        key="freecell_minimize_moves",
        help="Keep searching after the first solution to find the fewest moves. This can take a very long time on Freecell; stop at any prompt to keep the best solution found so far.",
    )

    job = st.session_state.get("freecell_solver_job")
    if st.button("Solve Freecell", type="primary", disabled=job is not None):
        try:
            target_card = None
            target_ranks = None
            goal_text = None
            if one_card:
                try:
                    target_card = Card(target_text)
                except ValueError as e:
                    raise ValueError(f"Target card: {e}")
                goal_text = f"{target_card.display} on the foundation"
            elif by_ranks:
                tokens = [t.strip() for t in ranks_text.replace(";", ",").split(",") if t.strip()]
                if not tokens:
                    raise ValueError("Ranks: enter at least one rank, e.g. 6 or 6, 8, K.")
                try:
                    target_ranks = sorted({Card(t + "S").value for t in tokens})
                except ValueError:
                    raise ValueError("Ranks: use A, 2-9, T, J, Q or K separated by commas, e.g. 6, 8, K.")
                names = ", ".join(Card(f"{'A23456789TJQK'[v - 1]}S").rank for v in target_ranks)
                goal_text = f"{rank_count} of each rank ({names}) on the foundations"
            # parse_input splits 28/24; Freecell needs all 52
            board_cards, stock_cards = parse_input(card_input)
            deck_cards = board_cards + stock_cards
        except ValueError as e:
            st.error(f"**Input Error:** {e}")
        except Exception as e:
            st.error("**An unexpected error occurred.**")
            st.exception(e)
        else:
            st.session_state.pop("freecell_solver_result", None)
            search_control = SearchControl()
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _run_freecell_search,
                deck_cards,
                minimize_moves,
                search_control,
                target_card,
                move_limit,
                target_ranks,
                rank_count,
            )
            started_at = monotonic()
            st.session_state.freecell_solver_job = {
                "minimize_moves": minimize_moves,
                "goal_text": goal_text,
                "move_limit": move_limit,
                "future": future,
                "executor": executor,
                "search_control": search_control,
                "started_at": started_at,
                "interval_started_at": started_at,
                "resume_pending": False,
                "stop_requested": False,
                "stop_requested_at": None,
                "finished_at": None,
            }

    job = st.session_state.get("freecell_solver_job")
    if job is not None:
        @st.fragment(run_every="1s")
        def render_search_status():
            if job.get("result_rendered"):
                result = st.session_state.get("freecell_solver_result")
                if result is not None:
                    _display_freecell_result(st, result)
                return

            future = job["future"]
            search_control = job["search_control"]

            if future.done():
                job["executor"].shutdown(wait=False)
                if job["finished_at"] is None:
                    job["finished_at"] = monotonic()
                try:
                    sequence, is_solved, proven_optimal = future.result()
                except Exception as e:
                    st.session_state.pop("freecell_solver_job", None)
                    job["result_rendered"] = True
                    st.error("**An unexpected error occurred.**")
                    st.exception(e)
                    return

                elapsed_until = job["stop_requested_at"] or job["finished_at"]
                result = {
                    "sequence": sequence,
                    "is_solved": is_solved,
                    "elapsed": elapsed_until - job["started_at"],
                    "stopped": job["stop_requested"],
                    "minimize_moves": job["minimize_moves"],
                    "goal_text": job["goal_text"],
                    "move_limit": job["move_limit"],
                    "proven_optimal": proven_optimal,
                }
                st.session_state.freecell_solver_result = result
                st.session_state.pop("freecell_solver_job", None)
                job["result_rendered"] = True
                _display_freecell_result(st, result)
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
                    if st.button("Continue searching", key="fc_continue_search"):
                        job["interval_started_at"] = monotonic()
                        job["resume_pending"] = True
                        search_control.resume_requested.set()
                with stop_col:
                    if st.button("Stop and show best result", key="fc_stop_search"):
                        job["stop_requested"] = True
                        job["stop_requested_at"] = monotonic()
                        search_control.stop_requested.set()
                        search_control.resume_requested.set()
                return

            if (
                not search_control.pause_requested.is_set()
                and monotonic() - job["interval_started_at"] >= SEARCH_PROMPT_INTERVAL_SECONDS
            ):
                search_control.pause_requested.set()

            elapsed_total = int(monotonic() - job["started_at"])
            label = "Pausing at the next search checkpoint" if search_control.pause_requested.is_set() else "Searching for a solution"
            st.status(f"{label}... {elapsed_total}s elapsed", state="running", expanded=False)

        render_search_status()
    elif "freecell_solver_result" in st.session_state:
        _display_freecell_result(st, st.session_state.freecell_solver_result)

if __name__ == "__main__":
    main()