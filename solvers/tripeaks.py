from solvers.search_control import SearchStopped

DEFAULT_MAX_STATES = 2_000_000


class TriPeaksBoard:
    def __init__(self):
        self.covered_by = {i: [] for i in range(28)}
        overlapping_links = {
            0: [3, 4], 1: [5, 6], 2: [7, 8],
            3: [9, 10], 4: [10, 11], 5: [12, 13], 6: [13, 14], 7: [15, 16], 8: [16, 17],
            9: [18, 19], 10: [19, 20], 11: [20, 21],
            12: [21, 22], 13: [22, 23], 14: [23, 24],
            15: [24, 25], 16: [25, 26], 17: [26, 27],
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
        if idx < 3:
            return 1, idx + 1
        if idx < 9:
            return 2, idx - 2
        if idx < 18:
            return 3, idx - 8
        if idx < 28:
            return 4, idx - 17
        return -1, -1


class TriPeaksSolver:
    def __init__(
        self,
        board_cards,
        stock_cards,
        board_layout,
        wrap_enabled=True,
        minimize_draws=False,
        search_control=None,
        max_states=DEFAULT_MAX_STATES,
    ):
        self.board_cards = board_cards
        self.initial_waste = stock_cards[0]
        self.stock_cards = stock_cards[1:]
        self.layout = board_layout
        self.wrap_enabled = wrap_enabled
        self.minimize_draws = minimize_draws
        self.search_control = search_control
        self.max_states = max_states
        self.max_draws = len(self.stock_cards)

        self.best_sequence = []
        self._best_moves = []
        self.max_cleared = 0
        self.min_stock_used = float("inf")
        self.visited_states = set()
        self.states_visited = 0
        self.state_limit_reached = False

        # Precompute value tables and adjacency bitmasks per waste rank (1..13)
        self._board_vals = tuple(c.value for c in board_cards)
        self._stock_vals = tuple(c.value for c in self.stock_cards)
        adj_masks = [0] * 14
        for wv in range(1, 14):
            mask = 0
            for idx, bv in enumerate(self._board_vals):
                diff = abs(wv - bv)
                if diff == 1 or (wrap_enabled and diff == 12):
                    mask |= 1 << idx
            adj_masks[wv] = mask
        self._adj_masks = tuple(adj_masks)

    def solve(self):
        self.best_sequence = []
        self._best_moves = []
        self.max_cleared = 0
        self.min_stock_used = float("inf")
        self.states_visited = 0
        self.state_limit_reached = False
        self.visited_states.clear()

        initial_board_mask = (1 << 28) - 1
        initial_free_mask = self.layout.initial_free_mask
        initial_waste_val = self.initial_waste.value

        try:
            if self.minimize_draws:
                self._solve_layered_bfs(initial_board_mask, initial_free_mask, initial_waste_val)
            else:
                self.max_draws = len(self.stock_cards)
                self._dfs(initial_board_mask, initial_free_mask, initial_waste_val, 0, [], 0)
        except SearchStopped:
            pass

        self.best_sequence = self._format_moves(self._best_moves)
        return self.best_sequence, self.max_cleared

    def _format_moves(self, moves):
        if self.min_stock_used == float("inf") and not moves:
            return []
        formatted = [f"Start: Waste pile is {self.initial_waste.display}"]
        row_pos = self.layout.row_pos
        board_cards = self.board_cards
        stock_cards = self.stock_cards
        for code in moves:
            if code >= 0:
                row, pos = row_pos[code]
                formatted.append(
                    f"Play {board_cards[code].display} from board (Row {row}, Position {pos})"
                )
            else:
                draw_idx = -1 - code
                formatted.append(f"Draw {stock_cards[draw_idx].display} from stock")
        return formatted

    def _update_best(self, cards_cleared, stock_drawn, moves):
        if cards_cleared > self.max_cleared or (
            cards_cleared == self.max_cleared and stock_drawn < self.min_stock_used
        ):
            self.max_cleared = cards_cleared
            self.min_stock_used = stock_drawn
            self._best_moves = list(moves)
            if self.search_control is not None:
                self.search_control.update_progress(f"Best: {cards_cleared}/28 cleared")

    def _solve_layered_bfs(self, initial_board_mask, initial_free_mask, initial_waste_val):
        """Layered 0-1 BFS ordered by stock_drawn (0..23).

        Within each draw layer, all 0-cost board plays are explored once;
        across draw boundaries, states with identical board_mask collapse to one
        representative because drawing overwrites the waste card. Previous layer
        visited sets are discarded immediately to keep peak RAM small.
        """
        cover_mask = self.layout.cover_mask
        covers = self.layout.covers
        board_vals = self._board_vals
        stock_vals = self._stock_vals
        adj_masks = self._adj_masks
        control = self.search_control
        max_states = self.max_states
        num_stock = len(stock_vals)

        # parents[node_id] = (parent_id, move_code)
        parents = [(-1, 0)]
        best_node_id = 0
        self.max_cleared = 0
        self.min_stock_used = 0

        def reconstruct(node_id):
            seq = []
            cur = node_id
            while cur > 0:
                parent_id, move_code = parents[cur]
                seq.append(move_code)
                cur = parent_id
            seq.reverse()
            return seq

        # Current layer seeds: list of (board_mask, free_mask, waste_val, node_id, cleared)
        current_seeds = [(initial_board_mask, initial_free_mask, initial_waste_val, 0, 0)]
        total_states = 0

        try:
            for stock_drawn in range(num_stock + 1):
                if not current_seeds:
                    break

                layer_visited = set()
                # Collapse identical board_masks before advancing to the next stock draw
                next_board_seeds = {}
                stack = current_seeds

                while stack:
                    board_mask, free_mask, waste_val, node_id, cleared = stack.pop()

                    if cleared > self.max_cleared or (
                        cleared == self.max_cleared and stock_drawn < self.min_stock_used
                    ):
                        self.max_cleared = cleared
                        self.min_stock_used = stock_drawn
                        best_node_id = node_id
                        if control is not None:
                            control.update_progress(f"Best: {cleared}/28 cleared")

                    if cleared == 28:
                        self._best_moves = reconstruct(node_id)
                        self.states_visited = total_states
                        self.visited_states = layer_visited
                        return True

                    if control is not None:
                        control.checkpoint()

                    layer_key = (board_mask << 4) | waste_val
                    if layer_key in layer_visited:
                        continue
                    layer_visited.add(layer_key)
                    total_states += 1
                    if max_states is not None and total_states >= max_states:
                        self.state_limit_reached = True
                        raise SearchStopped

                    if stock_drawn < num_stock and board_mask not in next_board_seeds:
                        next_board_seeds[board_mask] = (free_mask, node_id, cleared)

                    candidates = free_mask & adj_masks[waste_val]
                    while candidates:
                        lsb = candidates & -candidates
                        idx = lsb.bit_length() - 1
                        candidates ^= lsb

                        new_mask = board_mask ^ lsb
                        new_free = free_mask ^ lsb
                        for cov in covers[idx]:
                            if (new_mask & (1 << cov)) and (new_mask & cover_mask[cov]) == 0:
                                new_free |= 1 << cov

                        child_val = board_vals[idx]
                        child_key = (new_mask << 4) | child_val
                        if child_key not in layer_visited:
                            child_id = len(parents)
                            parents.append((node_id, idx))
                            stack.append((new_mask, new_free, child_val, child_id, cleared + 1))

                self.states_visited = total_states
                self.visited_states = layer_visited

                if stock_drawn < num_stock:
                    next_waste_val = stock_vals[stock_drawn]
                    draw_code = -1 - stock_drawn
                    next_seeds = []
                    for b_mask, (f_mask, n_id, clr) in next_board_seeds.items():
                        draw_node_id = len(parents)
                        parents.append((n_id, draw_code))
                        next_seeds.append((b_mask, f_mask, next_waste_val, draw_node_id, clr))
                    current_seeds = next_seeds
                else:
                    current_seeds = []
        finally:
            self.states_visited = total_states
            if not self._best_moves and best_node_id >= 0:
                self._best_moves = reconstruct(best_node_id)

        return False

    def _dfs(self, board_mask, free_mask, waste_val, stock_drawn, current_path, cards_cleared):
        if cards_cleared > self.max_cleared or (
            cards_cleared == self.max_cleared and stock_drawn < self.min_stock_used
        ):
            self.max_cleared = cards_cleared
            self.min_stock_used = stock_drawn
            self._best_moves = list(current_path)
            if self.search_control is not None:
                self.search_control.update_progress(f"Best: {cards_cleared}/28 cleared")

        if cards_cleared == 28:
            return True

        if self.search_control is not None:
            self.search_control.checkpoint()

        state_key = (board_mask << 9) | (waste_val << 5) | stock_drawn
        visited = self.visited_states
        if state_key in visited:
            return False
        visited.add(state_key)
        self.states_visited = len(visited)
        if self.max_states is not None and self.states_visited >= self.max_states:
            self.state_limit_reached = True
            raise SearchStopped

        cover_mask = self.layout.cover_mask
        covers = self.layout.covers
        board_vals = self._board_vals

        candidates = free_mask & self._adj_masks[waste_val]
        if candidates:
            # Build (unblocked_count, idx, new_mask, new_free) only for playable cards
            moves = []
            while candidates:
                lsb = candidates & -candidates
                idx = lsb.bit_length() - 1
                candidates ^= lsb

                new_mask = board_mask ^ lsb
                new_free = free_mask ^ lsb
                unblocked = 0
                for cov in covers[idx]:
                    if (new_mask & (1 << cov)) and (new_mask & cover_mask[cov]) == 0:
                        new_free |= 1 << cov
                        unblocked += 1
                moves.append((unblocked, idx, new_mask, new_free))

            if len(moves) > 1:
                moves.sort(key=lambda item: item[0], reverse=True)

            for _, idx, new_mask, new_free in moves:
                current_path.append(idx)
                if self._dfs(new_mask, new_free, board_vals[idx], stock_drawn, current_path, cards_cleared + 1):
                    return True
                current_path.pop()

        if stock_drawn < self.max_draws:
            current_path.append(-1 - stock_drawn)
            if self._dfs(
                board_mask,
                free_mask,
                self._stock_vals[stock_drawn],
                stock_drawn + 1,
                current_path,
                cards_cleared,
            ):
                return True
            current_path.pop()

        return False
