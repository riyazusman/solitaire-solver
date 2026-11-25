from solvers.search_control import SearchStopped

DEFAULT_MAX_STATES = 2_000_000


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
            15: [21, 22], 16: [22, 23], 17: [23, 24], 18: [24, 25], 19: [25, 26], 20: [26, 27],
        }

        for covered_card, covering_cards in overlapping_links.items():
            self.covered_by[covered_card].extend(covering_cards)

        # Precomputed O(1) bitmask tables
        self.cover_mask = tuple(
            sum(1 << c for c in self.covered_by[i]) for i in range(28)
        )
        covers_lists = [[] for _ in range(28)]
        for covered_card, covering_cards in self.covered_by.items():
            for cov in covering_cards:
                covers_lists[cov].append(covered_card)
        self.covers = tuple(tuple(lst) for lst in covers_lists)
        self.initial_free_mask = sum(
            1 << i for i in range(28) if self.cover_mask[i] == 0
        )
        self.row_pos = tuple(self.get_row_and_position(i) for i in range(28))

    def get_initial_free_cards(self):
        return [i for i, covers in self.covered_by.items() if not covers]

    def get_row_and_position(self, idx):
        if idx == 0:
            return 1, 1
        if idx < 3:
            return 2, idx
        if idx < 6:
            return 3, idx - 2
        if idx < 10:
            return 4, idx - 5
        if idx < 15:
            return 5, idx - 9
        if idx < 21:
            return 6, idx - 14
        if idx < 28:
            return 7, idx - 20
        return -1, -1


class PyramidSolver:
    def __init__(
        self,
        board_cards,
        stock_cards,
        board_layout,
        minimize_draws=False,
        max_passes=1,
        search_control=None,
        max_states=DEFAULT_MAX_STATES,
        strict_waste=True,
    ):
        self.board_cards = board_cards
        self.stock_cards = stock_cards
        self.layout = board_layout
        self.minimize_draws = minimize_draws
        self.max_passes = max_passes  # None means unlimited passes
        self.search_control = search_control
        self.max_states = max_states
        self.strict_waste = strict_waste
        self.max_draws = (
            float("inf") if self.max_passes is None else len(self.stock_cards) * self.max_passes
        )

        self.best_sequence = []
        self._best_moves = []
        self.max_cleared = 0
        self.min_stock_used = float("inf")
        self.visited_states = {}
        self.states_visited = 0
        self.state_limit_reached = False
        self._require_full_clear = False

        # Precompute values and rank bitmasks for O(1) complement dead-end pruning
        self._board_vals = tuple(c.value for c in board_cards)
        self._stock_vals = tuple(c.value for c in stock_cards)
        self._board_king_mask = sum(
            1 << i for i, v in enumerate(self._board_vals) if v == 13
        )
        self._board_rank_masks = tuple(
            sum(1 << i for i, v in enumerate(self._board_vals) if v == r)
            for r in range(14)
        )
        self._stock_rank_masks = tuple(
            sum(1 << i for i, v in enumerate(self._stock_vals) if v == r)
            for r in range(14)
        )

    def solve(self):
        self.best_sequence = []
        self._best_moves = []
        self.max_cleared = 0
        self.min_stock_used = float("inf")
        self.states_visited = 0
        self.state_limit_reached = False
        self._require_full_clear = False

        initial_board_mask = (1 << 28) - 1
        initial_stock = tuple(range(len(self.stock_cards)))
        initial_stock_alive = (1 << len(self.stock_cards)) - 1
        initial_drawn = None
        initial_waste = ()

        self.max_draws = (
            float("inf") if self.max_passes is None else len(self.stock_cards) * self.max_passes
        )
        self.visited_states.clear()
        try:
            won = self._dfs(
                initial_board_mask,
                initial_stock,
                initial_drawn,
                initial_waste,
                0,
                1,
                [],
                0,
                initial_stock_alive,
            )

            if won and self.minimize_draws and self.min_stock_used > 0:
                self._require_full_clear = True
                lo, hi = 0, self.min_stock_used - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    self.max_draws = mid
                    self.visited_states.clear()
                    if self._dfs(
                        initial_board_mask,
                        initial_stock,
                        initial_drawn,
                        initial_waste,
                        0,
                        1,
                        [],
                        0,
                        initial_stock_alive,
                    ):
                        # Found a full clear with <= mid draws (min_stock_used updated)
                        hi = self.min_stock_used - 1
                    else:
                        lo = mid + 1
        except SearchStopped:
            pass

        self.best_sequence = self._format_moves(self._best_moves)
        return self.best_sequence, self.max_cleared

    def _format_moves(self, moves):
        """Convert structured move tuples into human-readable move descriptions."""
        formatted = []
        row_pos = self.layout.row_pos
        board_cards = self.board_cards
        stock_cards = self.stock_cards
        i = 0
        n = len(moves)
        while i < n:
            move = moves[i]
            kind = move[0]
            if kind == "draw" and i + 1 < n:
                drawn_idx = move[1]
                nxt = moves[i + 1]
                nxt_kind = nxt[0]
                if nxt_kind == "play_drawn_king" and nxt[1] == drawn_idx:
                    formatted.append(f"Draw and immediately play {stock_cards[drawn_idx].display}")
                    i += 2
                    continue
                if nxt_kind == "pair_board_drawn" and nxt[2] == drawn_idx:
                    b_idx = nxt[1]
                    row, pos = row_pos[b_idx]
                    formatted.append(
                        f"Draw {stock_cards[drawn_idx].display} and immediately pair with "
                        f"{board_cards[b_idx].display} from board (Row {row}, Pos {pos})"
                    )
                    i += 2
                    continue
                if nxt_kind == "pair_drawn_waste" and nxt[1] == drawn_idx:
                    sub_idx = nxt[2]
                    formatted.append(
                        f"Draw {stock_cards[drawn_idx].display} and immediately pair with "
                        f"{stock_cards[sub_idx].display} in waste"
                    )
                    i += 2
                    continue

            if kind == "board_king":
                idx = move[1]
                row, pos = row_pos[idx]
                formatted.append(f"Play {board_cards[idx].display} from board (Row {row}, Position {pos})")
            elif kind == "board_pair":
                idx1, idx2 = move[1], move[2]
                r1, p1 = row_pos[idx1]
                r2, p2 = row_pos[idx2]
                formatted.append(
                    f"Pair {board_cards[idx1].display} (Row {r1}, Pos {p1}) and "
                    f"{board_cards[idx2].display} (Row {r2}, Pos {p2})"
                )
            elif kind in ("play_drawn_king", "play_waste_king"):
                s_idx = move[1]
                formatted.append(f"Play {stock_cards[s_idx].display} from waste")
            elif kind in ("pair_board_drawn", "pair_board_waste"):
                b_idx, s_idx = move[1], move[2]
                row, pos = row_pos[b_idx]
                formatted.append(
                    f"Pair {board_cards[b_idx].display} from board (Row {row}, Pos {pos}) and "
                    f"{stock_cards[s_idx].display} from waste"
                )
            elif kind in ("pair_drawn_waste", "pair_waste_waste"):
                d_idx, w_idx = move[1], move[2]
                formatted.append(f"Pair {stock_cards[d_idx].display} and {stock_cards[w_idx].display} in waste")
            elif kind == "draw":
                s_idx = move[1]
                formatted.append(f"Draw {stock_cards[s_idx].display} from stock")
            elif kind == "recycle":
                next_pass = move[1]
                pass_label = f"Pass {next_pass}/∞" if self.max_passes is None else f"Pass {next_pass}/{self.max_passes}"
                formatted.append(f"Recycle waste pile to stock ({pass_label})")
            i += 1
        return formatted

    def _get_free_mask(self, board_mask):
        cover_mask = self.layout.cover_mask
        free_mask = 0
        rem = board_mask
        while rem:
            lsb = rem & -rem
            idx = lsb.bit_length() - 1
            rem ^= lsb
            if (board_mask & cover_mask[idx]) == 0:
                free_mask |= lsb
        return free_mask

    def _max_clearable_upper_bound(self, board_mask, stock_alive_mask):
        """Sound upper bound on board cards clearable via rank-complement counts.

        Every remaining board card of rank r (1..12) must pair with a remaining
        card of rank (13 - r) from board/stock/drawn/waste. Any excess board cards
        of rank r over total remaining cards of rank (13 - r) can never be cleared.
        """
        b_masks = self._board_rank_masks
        s_masks = self._stock_rank_masks
        deficit = 0
        for r in range(1, 7):
            c = 13 - r
            br = (board_mask & b_masks[r]).bit_count()
            bc = (board_mask & b_masks[c]).bit_count()
            if br == 0 and bc == 0:
                continue
            tr = br + (stock_alive_mask & s_masks[r]).bit_count()
            tc = bc + (stock_alive_mask & s_masks[c]).bit_count()
            if br > tc:
                deficit += br - tc
            elif bc > tr:
                deficit += bc - tr
        return 28 - deficit

    def _unblock_score(self, board_mask, removed_mask):
        """Count how many covered board cards become uncovered after removing removed_mask."""
        new_mask = board_mask & ~removed_mask
        cover_mask = self.layout.cover_mask
        covers = self.layout.covers
        seen = 0
        count = 0
        rem = removed_mask
        while rem:
            lsb = rem & -rem
            idx = lsb.bit_length() - 1
            rem ^= lsb
            for cov in covers[idx]:
                bit = 1 << cov
                if not (seen & bit) and (new_mask & bit) and (new_mask & cover_mask[cov]) == 0:
                    seen |= bit
                    count += 1
        return count

    def _dfs(
        self,
        board_mask,
        stock_pile,
        drawn_idx,
        waste_pile,
        total_draws,
        passes,
        current_path,
        cards_cleared,
        stock_alive_mask=None,
    ):
        if stock_alive_mask is None:
            mask = 0
            for s_idx in stock_pile:
                mask |= 1 << s_idx
            for s_idx in waste_pile:
                mask |= 1 << s_idx
            if drawn_idx is not None:
                mask |= 1 << drawn_idx
            stock_alive_mask = mask

        board_vals = self._board_vals
        stock_vals = self._stock_vals
        cover_mask = self.layout.cover_mask
        covers = self.layout.covers
        king_mask = self._board_king_mask

        # Strict-dominance greedy removal: any exposed King (board, drawn slot, or waste top)
        # can never pair with another card, costs 0 draws, and strictly improves the state.
        auto_pushed = 0
        free_mask = self._get_free_mask(board_mask)
        while True:
            progressed = False
            free_kings = free_mask & king_mask
            while free_kings:
                lsb = free_kings & -free_kings
                idx = lsb.bit_length() - 1
                free_kings ^= lsb
                board_mask ^= lsb
                free_mask ^= lsb
                cards_cleared += 1
                current_path.append(("board_king", idx))
                auto_pushed += 1
                for cov in covers[idx]:
                    bit = 1 << cov
                    if (board_mask & bit) and (board_mask & cover_mask[cov]) == 0:
                        free_mask |= bit
                        if king_mask & bit:
                            free_kings |= bit
                progressed = True

            if drawn_idx is not None and stock_vals[drawn_idx] == 13:
                current_path.append(("play_drawn_king", drawn_idx))
                auto_pushed += 1
                stock_alive_mask &= ~(1 << drawn_idx)
                drawn_idx = None
                progressed = True

            while waste_pile and stock_vals[waste_pile[-1]] == 13:
                k_idx = waste_pile[-1]
                waste_pile = waste_pile[:-1]
                current_path.append(("play_waste_king", k_idx))
                auto_pushed += 1
                stock_alive_mask &= ~(1 << k_idx)
                progressed = True

            if not progressed:
                break

        try:
            if cards_cleared > self.max_cleared or (
                cards_cleared == self.max_cleared and total_draws < self.min_stock_used
            ):
                self.max_cleared = cards_cleared
                self.min_stock_used = total_draws
                self._best_moves = list(current_path)
                if self.search_control is not None:
                    self.search_control.update_progress(f"Best: {cards_cleared}/28 cleared")

            if cards_cleared == 28:
                return True

            if self.search_control is not None:
                self.search_control.checkpoint()

            # Rank-complement dead-end pruning
            upper_bound = self._max_clearable_upper_bound(board_mask, stock_alive_mask)
            if self._require_full_clear:
                if upper_bound < 28:
                    return False
            else:
                if upper_bound < self.max_cleared or (
                    upper_bound == self.max_cleared and total_draws >= self.min_stock_used
                ):
                    return False

            state_key = (
                (board_mask, stock_pile, drawn_idx, waste_pile)
                if self.max_passes is None
                else (board_mask, stock_pile, drawn_idx, waste_pile, passes)
            )
            previous_draws = self.visited_states.get(state_key)
            if previous_draws is not None and previous_draws <= total_draws:
                return False
            self.visited_states[state_key] = total_draws
            self.states_visited = len(self.visited_states)
            if self.max_states is not None and self.states_visited >= self.max_states:
                self.state_limit_reached = True
                raise SearchStopped

            # Extract free board indices
            free_indices = []
            rem = free_mask
            while rem:
                lsb = rem & -rem
                free_indices.append(lsb.bit_length() - 1)
                rem ^= lsb

            # 1. Pair Board + Board (ordered by cards unblocked, descending)
            n_free = len(free_indices)
            board_pairs = []
            for i in range(n_free):
                idx1 = free_indices[i]
                v1 = board_vals[idx1]
                need = 13 - v1
                for j in range(i + 1, n_free):
                    idx2 = free_indices[j]
                    if board_vals[idx2] == need:
                        pair_mask = (1 << idx1) | (1 << idx2)
                        score = self._unblock_score(board_mask, pair_mask)
                        board_pairs.append((score, idx1, idx2, pair_mask))
            if len(board_pairs) > 1:
                board_pairs.sort(key=lambda item: item[0], reverse=True)

            for _, idx1, idx2, pair_mask in board_pairs:
                current_path.append(("board_pair", idx1, idx2))
                if self._dfs(
                    board_mask ^ pair_mask,
                    stock_pile,
                    drawn_idx,
                    waste_pile,
                    total_draws,
                    passes,
                    current_path,
                    cards_cleared + 2,
                    stock_alive_mask,
                ):
                    return True
                current_path.pop()

            # 2. Moves using the active drawn stock card (face-up draw slot)
            if drawn_idx is not None:
                drawn_val = stock_vals[drawn_idx]
                drawn_need = 13 - drawn_val
                next_alive = stock_alive_mask & ~(1 << drawn_idx)

                # 2a. Pair Board + Draw slot (ordered by cards unblocked)
                bd_moves = [
                    (self._unblock_score(board_mask, 1 << idx), idx)
                    for idx in free_indices
                    if board_vals[idx] == drawn_need
                ]
                if len(bd_moves) > 1:
                    bd_moves.sort(key=lambda item: item[0], reverse=True)

                for _, idx in bd_moves:
                    current_path.append(("pair_board_drawn", idx, drawn_idx))
                    if self._dfs(
                        board_mask ^ (1 << idx),
                        stock_pile,
                        None,
                        waste_pile,
                        total_draws,
                        passes,
                        current_path,
                        cards_cleared + 1,
                        next_alive,
                    ):
                        return True
                    current_path.pop()

                # 2b. Pair active Draw slot + top of Waste pile
                if waste_pile:
                    top_waste_idx = waste_pile[-1]
                    if stock_vals[top_waste_idx] == drawn_need:
                        new_waste = waste_pile[:-1]
                        current_path.append(("pair_drawn_waste", drawn_idx, top_waste_idx))
                        if self._dfs(
                            board_mask,
                            stock_pile,
                            None,
                            new_waste,
                            total_draws,
                            passes,
                            current_path,
                            cards_cleared,
                            next_alive & ~(1 << top_waste_idx),
                        ):
                            return True
                        current_path.pop()

            # 3. Pair Board + top of Waste pile
            if waste_pile:
                top_waste_idx = waste_pile[-1]
                waste_need = 13 - stock_vals[top_waste_idx]
                bw_moves = [
                    (self._unblock_score(board_mask, 1 << idx), idx)
                    for idx in free_indices
                    if board_vals[idx] == waste_need
                ]
                if len(bw_moves) > 1:
                    bw_moves.sort(key=lambda item: item[0], reverse=True)

                if bw_moves:
                    new_waste = waste_pile[:-1]
                    next_alive = stock_alive_mask & ~(1 << top_waste_idx)
                    for _, idx in bw_moves:
                        current_path.append(("pair_board_waste", idx, top_waste_idx))
                        if self._dfs(
                            board_mask ^ (1 << idx),
                            stock_pile,
                            drawn_idx,
                            new_waste,
                            total_draws,
                            passes,
                            current_path,
                            cards_cleared + 1,
                            next_alive,
                        ):
                            return True
                        current_path.pop()

                # 3b. Optional relaxed variant: pair top two cards buried in waste
                if not self.strict_waste and drawn_idx is None and len(waste_pile) >= 2:
                    sub_waste_idx = waste_pile[-2]
                    if stock_vals[sub_waste_idx] == waste_need:
                        new_waste = waste_pile[:-2]
                        next_alive = stock_alive_mask & ~(1 << top_waste_idx) & ~(1 << sub_waste_idx)
                        current_path.append(("pair_waste_waste", top_waste_idx, sub_waste_idx))
                        if self._dfs(
                            board_mask,
                            stock_pile,
                            None,
                            new_waste,
                            total_draws,
                            passes,
                            current_path,
                            cards_cleared,
                            next_alive,
                        ):
                            return True
                        current_path.pop()

            # 4. Draw from Stock
            if stock_pile and total_draws < self.max_draws:
                next_drawn_idx = stock_pile[0]
                new_stock = stock_pile[1:]
                new_waste = waste_pile + (drawn_idx,) if drawn_idx is not None else waste_pile
                current_path.append(("draw", next_drawn_idx))
                if self._dfs(
                    board_mask,
                    new_stock,
                    next_drawn_idx,
                    new_waste,
                    total_draws + 1,
                    passes,
                    current_path,
                    cards_cleared,
                    stock_alive_mask,
                ):
                    return True
                current_path.pop()

            # 5. Recycle Waste to Stock
            if not stock_pile and (self.max_passes is None or passes < self.max_passes):
                full_waste = waste_pile + (drawn_idx,) if drawn_idx is not None else waste_pile
                if full_waste:
                    current_path.append(("recycle", passes + 1))
                    if self._dfs(
                        board_mask,
                        full_waste,
                        None,
                        (),
                        total_draws,
                        passes + 1,
                        current_path,
                        cards_cleared,
                        stock_alive_mask,
                    ):
                        return True
                    current_path.pop()

            return False
        finally:
            if auto_pushed:
                del current_path[-auto_pushed:]
