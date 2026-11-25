import heapq

from solvers.models import Card
from solvers.search_control import SearchStopped

DEFAULT_MAX_STATES = 1_000_000

# Precomputed integer card tables: cid = suit_idx * 13 + (rank - 1), S=0 H=1 D=2 C=3
_SUITS = ("S", "H", "D", "C")
_SUIT_INDEX = {"S": 0, "H": 1, "D": 2, "C": 3}
_SUIT_SYMBOLS = ("♠️", "♥️", "♦️", "♣️")
_RANK_STRS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")

_CARD_RANK = tuple((cid % 13) + 1 for cid in range(52))
_CARD_SUIT = tuple(cid // 13 for cid in range(52))
_CARD_RED = tuple((cid // 13) in (1, 2) for cid in range(52))
_CARD_DISPLAY = tuple(f"{_RANK_STRS[cid % 13]}{_SUIT_SYMBOLS[cid // 13]}" for cid in range(52))
_OPPOSITE_SUITS = ((1, 2), (0, 3), (0, 3), (1, 2))


def _format_move(move) -> str:
    """Format a compact integer move tuple into human-readable text on demand."""
    kind = move[0]
    if kind == 0:  # cascade -> foundation
        _, src, cid, is_auto = move
        suffix = " (auto)" if is_auto else ""
        return f"Move {_CARD_DISPLAY[cid]} from Cascade {src + 1} to Foundation{suffix}"
    if kind == 1:  # freecell -> foundation
        _, cid, is_auto = move
        suffix = " (auto)" if is_auto else ""
        return f"Move {_CARD_DISPLAY[cid]} from Freecell to Foundation{suffix}"
    if kind == 2:  # cascade -> freecell
        _, src, cid = move
        return f"Move {_CARD_DISPLAY[cid]} from Cascade {src + 1} to Freecell"
    if kind == 3:  # freecell -> cascade
        _, dest, cid, target_cid = move
        if target_cid < 0:
            return f"Move {_CARD_DISPLAY[cid]} from Freecell to empty Cascade {dest + 1}"
        return f"Move {_CARD_DISPLAY[cid]} from Freecell to Cascade {dest + 1} on {_CARD_DISPLAY[target_cid]}"
    if kind == 5:  # foundation -> cascade
        _, dest, cid, target_cid = move
        if target_cid < 0:
            return f"Move {_CARD_DISPLAY[cid]} from Foundation to empty Cascade {dest + 1}"
        return f"Move {_CARD_DISPLAY[cid]} from Foundation to Cascade {dest + 1} on {_CARD_DISPLAY[target_cid]}"
    # kind == 4: cascade -> cascade
    _, src, dest, cards, target_cid = move
    count = len(cards)
    if count == 1:
        what = _CARD_DISPLAY[cards[0]]
    else:
        what = f"{count} cards ({_CARD_DISPLAY[cards[0]]} to {_CARD_DISPLAY[cards[-1]]})"
    where = (
        f"empty Cascade {dest + 1}"
        if target_cid < 0
        else f"Cascade {dest + 1} on {_CARD_DISPLAY[target_cid]}"
    )
    return f"Move {what} from Cascade {src + 1} to {where}"


class FreecellState:
    def __init__(self, deck_cards=None):
        self.cascades = [[] for _ in range(8)]
        self.freecells = []
        # highest rank placed per suit (0-13)
        self.foundations = {"S": 0, "H": 0, "D": 0, "C": 0}

        if deck_cards:
            if len(deck_cards) != 52:
                raise ValueError("Freecell requires exactly 52 cards.")
            for i, card in enumerate(deck_cards):
                self.cascades[i % 8].append(card)

    def is_solved(self):
        return all(rank == 13 for rank in self.foundations.values())

    def get_state_key(self):
        frozen_cascades = tuple(sorted(tuple(c.int_id for c in cascade) for cascade in self.cascades))
        frozen_freecells = tuple(sorted(c.int_id for c in self.freecells))
        f = self.foundations
        frozen_foundations = (f["S"], f["H"], f["D"], f["C"])
        return (frozen_cascades, frozen_freecells, frozen_foundations)


class FreecellSolver:
    def __init__(
        self,
        initial_state,
        search_control=None,
        minimize_moves=False,
        target_card=None,
        move_limit=None,
        target_ranks=None,
        rank_count=None,
        max_states=DEFAULT_MAX_STATES,
        allow_foundation_to_cascade=False,
        atomic_move_count=False,
    ):
        """
        target_card: get only that card to its foundation (auto-play off, each action is one move).
        target_ranks/rank_count: get rank_count (default 4) cards of each of those ranks (values 1-13) up.
        move_limit: maximum number of moves, for any goal.
        allow_foundation_to_cascade: allow moving top foundation cards back down onto cascades.
        atomic_move_count: count each card in a multi-card supermove as 1 move instead of 1 move per stack.
        """
        self.state = initial_state
        self.search_control = search_control
        self.minimize_moves = minimize_moves
        self.target_card = target_card
        self.move_limit = move_limit
        self.target_ranks = tuple(sorted(set(target_ranks))) if target_ranks else None
        self.rank_count = rank_count or 4
        self.whole_board = target_card is None and not self.target_ranks
        self.max_states = max_states
        self.allow_foundation_to_cascade = allow_foundation_to_cascade
        self.atomic_move_count = atomic_move_count

        self._target_suit = _SUIT_INDEX[target_card.suit] if target_card is not None else -1
        self._target_val = target_card.value if target_card is not None else 0
        self._top_rank = self.target_ranks[-1] if self.target_ranks else 0

        self.visited_states = {}
        self.states_visited = 0
        self.state_limit_reached = False
        self.winning_sequence = []
        self.best_move_cost = 0

    def _to_int_node(self, state):
        cascades = tuple(tuple(c.int_id for c in col) for col in state.cascades)
        freecells = tuple(c.int_id for c in state.freecells)
        f = state.foundations
        foundations = (f["S"], f["H"], f["D"], f["C"])
        return (cascades, freecells, foundations)

    def _goal_remaining(self, foundations=None):
        """Lower bound on moves still needed (0 only when the goal is met)."""
        if foundations is None:
            f = self.state.foundations
            found = (f["S"], f["H"], f["D"], f["C"])
        elif isinstance(foundations, dict):
            found = (foundations["S"], foundations["H"], foundations["D"], foundations["C"])
        else:
            found = foundations

        if self.whole_board:
            return 52 - (found[0] + found[1] + found[2] + found[3])
        if self.target_card is not None:
            rem = self._target_val - found[self._target_suit]
            return rem if rem > 0 else 0

        rc = self.rank_count
        have = [
            (found[0] >= r) + (found[1] >= r) + (found[2] >= r) + (found[3] >= r)
            for r in self.target_ranks
        ]
        unsat_ranks = [r for r, h in zip(self.target_ranks, have) if h < rc]
        if not unsat_ranks:
            return 0
        need = sum(rc - h for h in have if h < rc)
        inf = 10 ** 6
        best = [0] + [inf] * need
        for height in found:
            wanted = [r for r in unsat_ranks if r > height]
            if not wanted:
                continue
            options = [(0, 0)] + [(k, r - height) for k, r in enumerate(wanted, 1)]
            nxt = [inf] * (need + 1)
            for got in range(need + 1):
                base = best[got]
                if base >= inf:
                    continue
                for k, cost in options:
                    reached = got + k
                    if reached > need:
                        reached = need
                    cand = base + cost
                    if cand < nxt[reached]:
                        nxt[reached] = cand
            best = nxt
        res = best[need]
        return res if res > 1 else 1

    def solve(self):
        """
        Iterative best-first search. With minimize_moves, later passes weight path length more and
        prune states that cannot beat the best solution; the last pass is A*.
        proven is True when the search space was exhausted.
        Returns (path, solved, proven).
        """
        self.winning_sequence = []
        self.best_solution = None
        self.best_solution_cost = float("inf")
        self.best_partial = []
        self.best_found = -float("inf")
        self.states_visited = 0
        self.state_limit_reached = False
        self._initial_node = self._to_int_node(self.state)
        self._reexpand = self.minimize_moves or self.move_limit is not None
        proven = False
        passes = [(0.5, False)]
        if self.minimize_moves:
            passes += [(2.0, False), (6.0, False), (1.0, True)]

        try:
            for weight, final in passes:
                if self.best_solution is not None:
                    bound = self.best_solution_cost
                elif self.move_limit is not None:
                    bound = self.move_limit + 1
                else:
                    bound = float("inf")
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
        self.best_move_cost = self.best_solution_cost if solved else len(path)
        return path, solved, proven

    def _state_key(self, cascades, freecells, foundations):
        return (
            tuple(sorted(tuple(c) for c in cascades)),
            tuple(sorted(freecells)),
            tuple(foundations),
        )

    def _auto_foundation(self, cascades, freecells, foundations):
        """Move safe cards to foundations in-place; return compact move tuples."""
        moves = []
        rank_of = _CARD_RANK
        suit_of = _CARD_SUIT
        opp_of = _OPPOSITE_SUITS

        progress = True
        while progress:
            progress = False
            for i in range(8):
                col = cascades[i]
                if not col:
                    continue
                cid = col[-1]
                suit = suit_of[cid]
                rank = rank_of[cid]
                if foundations[suit] != rank - 1:
                    continue
                if rank > 2:
                    o0, o1 = opp_of[suit]
                    if foundations[o0] < rank - 1 or foundations[o1] < rank - 1:
                        continue
                col.pop()
                foundations[suit] += 1
                moves.append((0, i, cid, True))
                progress = True
                break
            if progress:
                continue
            for i, cid in enumerate(freecells):
                suit = suit_of[cid]
                rank = rank_of[cid]
                if foundations[suit] != rank - 1:
                    continue
                if rank > 2:
                    o0, o1 = opp_of[suit]
                    if foundations[o0] < rank - 1 or foundations[o1] < rank - 1:
                        continue
                freecells.pop(i)
                foundations[suit] += 1
                moves.append((1, cid, True))
                progress = True
                break
        return moves

    def _priority(self, cascades, freecells, foundations, depth, remaining, weight, final):
        """Lower is better. The final pass is A* (depth + remaining)."""
        if final:
            return depth + remaining

        rank_of = _CARD_RANK
        suit_of = _CARD_SUIT
        buried = 0
        disorder = 0

        if self.whole_board:
            for col in cascades:
                n = len(col)
                lowest_above = 99
                for pos, cid in enumerate(col):
                    r = rank_of[cid]
                    if r == foundations[suit_of[cid]] + 1:
                        buried += n - pos - 1
                    if r > lowest_above:
                        disorder += 1
                    elif r < lowest_above:
                        lowest_above = r
        elif self.target_card is not None:
            t_suit = self._target_suit
            t_val = self._target_val
            f_h = foundations[t_suit]
            for col in cascades:
                n = len(col)
                for pos, cid in enumerate(col):
                    if suit_of[cid] == t_suit:
                        r = rank_of[cid]
                        if f_h < r <= t_val:
                            buried += n - pos - 1
        else:
            top_rank = self._top_rank
            for col in cascades:
                n = len(col)
                for pos, cid in enumerate(col):
                    s = suit_of[cid]
                    r = rank_of[cid]
                    if foundations[s] < r <= top_rank:
                        buried += n - pos - 1

        return 10 * remaining + 3 * buried + disorder + 2 * len(freecells) + depth * weight

    def _generate_valid_moves(self, cascades, freecells, foundations):
        moves = []
        rank_of = _CARD_RANK
        suit_of = _CARD_SUIT
        red_of = _CARD_RED

        free_cells = 4 - len(freecells)
        empty_columns = sum(1 for c in cascades if not c)
        cap_non_empty = (free_cells + 1) << empty_columns
        cap_empty = (free_cells + 1) << (empty_columns - 1) if empty_columns > 0 else 0

        for i in range(8):
            cascade = cascades[i]
            col_len = len(cascade)
            if col_len == 0:
                continue
            card = cascade[-1]
            card_rank = rank_of[card]
            card_suit = suit_of[card]

            if foundations[card_suit] == card_rank - 1:
                moves.append((0, i, card, False))

            run = 1
            while run < col_len:
                upper = cascade[-run - 1]
                lower = cascade[-run]
                if red_of[upper] != red_of[lower] and rank_of[upper] == rank_of[lower] + 1:
                    run += 1
                else:
                    break

            empty_used = False
            for j in range(8):
                if i == j:
                    continue
                target = cascades[j]
                if not target:
                    if empty_used:
                        continue
                    empty_used = True
                    max_n = min(run, cap_empty, col_len - 1)
                    for n in range(1, max_n + 1):
                        moves.append((4, i, j, tuple(cascade[-n:]), -1))
                else:
                    target_card = target[-1]
                    n = rank_of[target_card] - card_rank
                    if 1 <= n <= run and n <= cap_non_empty:
                        bottom = cascade[-n]
                        if red_of[bottom] != red_of[target_card] and rank_of[bottom] == rank_of[target_card] - 1:
                            moves.append((4, i, j, tuple(cascade[-n:]), target_card))

            if free_cells > 0:
                moves.append((2, i, card))

        for i, card in enumerate(freecells):
            card_rank = rank_of[card]
            card_suit = suit_of[card]
            if foundations[card_suit] == card_rank - 1:
                moves.append((1, card, False))

            empty_used = False
            for j in range(8):
                target = cascades[j]
                if not target:
                    if empty_used:
                        continue
                    empty_used = True
                    moves.append((3, j, card, -1))
                else:
                    target_card = target[-1]
                    if red_of[target_card] != red_of[card] and rank_of[target_card] == card_rank + 1:
                        moves.append((3, j, card, target_card))

        if self.allow_foundation_to_cascade:
            for suit_idx in range(4):
                top_rank = foundations[suit_idx]
                if top_rank <= 0:
                    continue
                # Never pull back a card that auto-foundation would immediately push back up
                if self.whole_board:
                    if top_rank <= 2:
                        continue
                    o0, o1 = _OPPOSITE_SUITS[suit_idx]
                    if foundations[o0] >= top_rank - 1 and foundations[o1] >= top_rank - 1:
                        continue
                cid = suit_idx * 13 + (top_rank - 1)
                c_red = red_of[cid]
                empty_used = False
                for j in range(8):
                    target = cascades[j]
                    if not target:
                        if empty_used:
                            continue
                        empty_used = True
                        moves.append((5, j, cid, -1))
                    else:
                        target_card = target[-1]
                        if red_of[target_card] != c_red and rank_of[target_card] == top_rank + 1:
                            moves.append((5, j, cid, target_card))

        return moves

    def _apply_move(self, cascades, freecells, foundations, move):
        kind = move[0]
        if kind == 0:  # cascade -> foundation
            src, cid = move[1], move[2]
            cascades[src].pop()
            foundations[_CARD_SUIT[cid]] += 1
        elif kind == 1:  # freecell -> foundation
            cid = move[1]
            freecells.remove(cid)
            foundations[_CARD_SUIT[cid]] += 1
        elif kind == 2:  # cascade -> freecell
            src, cid = move[1], move[2]
            cascades[src].pop()
            freecells.append(cid)
        elif kind == 3:  # freecell -> cascade
            dest, cid = move[1], move[2]
            freecells.remove(cid)
            cascades[dest].append(cid)
        elif kind == 5:  # foundation -> cascade
            dest, cid = move[1], move[2]
            foundations[_CARD_SUIT[cid]] -= 1
            cascades[dest].append(cid)
        else:  # kind == 4: cascade -> cascade
            src, dest, cards = move[1], move[2], move[3]
            del cascades[src][-len(cards):]
            cascades[dest].extend(cards)

    def _move_cost(self, move):
        if self.atomic_move_count and move[0] == 4:
            return len(move[3])
        return 1

    def _search_pass(self, weight, final, best_len):
        """Returns True if the queue was exhausted."""
        self.visited_states.clear()

        # parents[i] = (parent_index, move_tuples, moves_so_far)
        parents = [(-1, (), 0)]
        counter = 0

        init_casc, init_fc, init_found = self._initial_node
        cascades = [list(c) for c in init_casc]
        freecells = list(init_fc)
        foundations = list(init_found)

        start_moves = self._auto_foundation(cascades, freecells, foundations) if self.whole_board else []
        start_depth = len(start_moves)
        start_remaining = self._goal_remaining(foundations)
        parents.append((0, tuple(start_moves), start_depth))
        start_key = self._state_key(cascades, freecells, foundations)
        start_node = (tuple(tuple(c) for c in cascades), tuple(freecells), tuple(foundations))

        def path_to(index):
            chunks = []
            while index > 0:
                chunks.append(parents[index][1])
                index = parents[index][0]
            return [_format_move(m) for chunk in reversed(chunks) for m in chunk]

        best_partial_index = None
        if start_remaining == 0:
            if start_depth < best_len:
                self.best_solution = path_to(1)
                self.best_solution_cost = start_depth
            elif self.move_limit is not None and not self.best_partial and self.best_solution is None:
                self.best_partial = [_format_move(m) for m in start_moves[: self.move_limit]]
            return True

        if start_depth > 0 and start_depth < best_len and -start_remaining > self.best_found:
            self.best_found = -start_remaining
            best_partial_index = 1

        if start_depth + start_remaining >= best_len:
            if best_partial_index is not None:
                self.best_partial = path_to(best_partial_index)
            elif start_depth > 0 and self.move_limit is not None and not self.best_partial and self.best_solution is None:
                self.best_partial = [_format_move(m) for m in start_moves[: self.move_limit]]
            return True

        start_priority = self._priority(
            cascades, freecells, foundations, start_depth, start_remaining, weight, final
        )
        heap = [(start_priority, counter, 1, start_node, start_key, start_remaining)]
        visited = self.visited_states
        visited[start_key] = start_depth
        control = self.search_control
        max_states = self.max_states
        reexpand = self._reexpand
        whole_board = self.whole_board

        try:
            while heap:
                if control is not None:
                    control.checkpoint()

                priority, _, index, node, key, node_rem = heapq.heappop(heap)
                depth = parents[index][2]
                if final and priority >= best_len:
                    return True
                if reexpand and depth > visited.get(key, depth):
                    continue
                if depth + node_rem >= best_len:
                    continue

                node_casc, node_fc, node_found = node
                cascades = [list(c) for c in node_casc]
                freecells = list(node_fc)
                foundations = list(node_found)

                for move in self._generate_valid_moves(cascades, freecells, foundations):
                    self._apply_move(cascades, freecells, foundations, move)
                    move_cost = self._move_cost(move)
                    if whole_board:
                        auto_moves = self._auto_foundation(cascades, freecells, foundations)
                        step_moves = (move, *auto_moves) if auto_moves else (move,)
                        step_cost = move_cost + len(auto_moves)
                    else:
                        step_moves = (move,)
                        step_cost = move_cost

                    child_depth = depth + step_cost
                    remaining = self._goal_remaining(foundations)
                    if child_depth + remaining < best_len:
                        child_key = self._state_key(cascades, freecells, foundations)
                        known = visited.get(child_key)
                        if known is None or (reexpand and child_depth < known):
                            visited[child_key] = child_depth
                            self.states_visited = len(visited)
                            parents.append((index, step_moves, child_depth))
                            child_index = len(parents) - 1

                            if -remaining > self.best_found:
                                self.best_found = -remaining
                                best_partial_index = child_index
                                if control is not None:
                                    if whole_board:
                                        control.update_progress(f"Best: {52 - remaining}/52 on foundations")
                                    else:
                                        control.update_progress(f"Best: {remaining} moves remaining")

                            if remaining == 0:
                                best_len = child_depth
                                self.best_solution = path_to(child_index)
                                self.best_solution_cost = child_depth
                                if not final:
                                    return False
                            else:
                                if max_states is not None and self.states_visited >= max_states:
                                    self.state_limit_reached = True
                                    raise SearchStopped
                                counter += 1
                                child_node = (
                                    tuple(tuple(c) for c in cascades),
                                    tuple(freecells),
                                    tuple(foundations),
                                )
                                child_prio = self._priority(
                                    cascades, freecells, foundations, child_depth, remaining, weight, final
                                )
                                heapq.heappush(
                                    heap,
                                    (child_prio, counter, child_index, child_node, child_key, remaining),
                                )

                    # Restore working lists to parent node
                    cascades = [list(c) for c in node_casc]
                    freecells = list(node_fc)
                    foundations = list(node_found)
            return True
        finally:
            self.states_visited = len(visited)
            if best_partial_index is not None:
                self.best_partial = path_to(best_partial_index)
