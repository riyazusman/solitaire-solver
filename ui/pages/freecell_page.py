import streamlit as st

from solvers.freecell import DEFAULT_MAX_STATES, FreecellSolver, FreecellState
from solvers.models import Card, parse_deck
from solvers.replay import replay_freecell
from ui.board_views import render_freecell_board, render_step_playback
from ui.deck_input import render_deck_input
from ui.job_runner import (
    cancel_other_page_jobs,
    render_search_settings,
    render_solver_job_area,
    start_solver_job,
)

JOB_KEY = "freecell_solver_job"
RESULT_KEY = "freecell_solver_result"
ERROR_KEY = "freecell_solver_error"


def _run_freecell_search(
    deck_cards,
    minimize_moves,
    target_card,
    move_limit,
    target_ranks,
    rank_count,
    allow_foundation_to_cascade,
    atomic_move_count,
    max_states,
    search_control,
):
    solver = FreecellSolver(
        initial_state=FreecellState(deck_cards),
        search_control=search_control,
        minimize_moves=minimize_moves,
        target_card=target_card,
        move_limit=move_limit,
        target_ranks=target_ranks,
        rank_count=rank_count,
        max_states=max_states,
        allow_foundation_to_cascade=allow_foundation_to_cascade,
        atomic_move_count=atomic_move_count,
    )
    sequence, is_solved, proven_optimal = solver.solve()
    return (
        sequence,
        is_solved,
        proven_optimal,
        solver.state_limit_reached,
        solver.states_visited,
        solver.best_move_cost,
    )


def _build_freecell_result(raw_result, search_control, metadata):
    sequence, is_solved, proven_optimal, state_limit_reached, states_visited, move_cost = raw_result
    return {
        "sequence": sequence,
        "is_solved": is_solved,
        "elapsed": search_control.active_elapsed,
        "stopped": search_control.stop_requested,
        "state_limit_reached": state_limit_reached,
        "states_visited": states_visited,
        "move_cost": move_cost,
        "atomic_move_count": metadata["atomic_move_count"],
        "minimize_moves": metadata["minimize_moves"],
        "goal_text": metadata["goal_text"],
        "move_limit": metadata["move_limit"],
        "proven_optimal": proven_optimal,
        "deck_cards": metadata["deck_cards"],
    }


def _display_freecell_result(st, result):
    sequence = result["sequence"]
    is_solved = result["is_solved"]
    elapsed = result["elapsed"]
    stopped = result["stopped"]
    state_limit_reached = result.get("state_limit_reached", False)
    minimize_moves = result["minimize_moves"]
    goal = result.get("goal_text")
    move_limit = result.get("move_limit")
    limit_text = f" within {move_limit} moves" if move_limit else ""

    st.divider()
    if state_limit_reached:
        st.warning(
            f"⚠️ Memory safety state limit reached after {elapsed:.2f}s. "
            + ("Showing best solution found so far:" if is_solved else "Showing closest progress found so far:")
        )
    elif goal is not None:
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

    if result.get("atomic_move_count"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Atomic Moves", value=result.get("move_cost", len(sequence)))
        with c2:
            st.metric(label="Total Steps", value=len(sequence))
        with c3:
            st.metric(label="States Explored", value=f"{result.get('states_visited', 0):,}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Total Moves", value=len(sequence))
        with c2:
            st.metric(label="States Explored", value=f"{result.get('states_visited', 0):,}")

    if not sequence:
        st.info("No valid moves could be made from the starting layout.")
        return

    text_tab, playback_tab = st.tabs(["📜 Move Sequence", "🎬 Step-by-Step Playback"])
    with text_tab:
        st.subheader("Move Sequence:")
        sequence_text = "".join(f"{step_num + 1}. {step}\n" for step_num, step in enumerate(sequence))
        st.code(sequence_text, language="text")

    with playback_tab:
        snapshots = replay_freecell(result["deck_cards"], sequence)
        render_step_playback(
            st,
            key_prefix="freecell",
            snapshots=snapshots,
            render_snapshot_fn=lambda snap: render_freecell_board(
                st, snap.cascades, snap.freecells, snap.foundations
            ),
        )


def main():
    cancel_other_page_jobs(st.session_state, JOB_KEY)
    job = st.session_state.get(JOB_KEY)
    is_running = job is not None

    st.title("🃏 Freecell Solitaire Solver")
    st.markdown("""
    Find the optimal sequence to clear a Freecell board.
    Enter 52 cards separated by commas, spaces, or newlines. The first 8 cards form the top row of the cascades, the next 8 form the second row, and so on.
    """)

    with st.expander("📐 Deal Structure Guide", expanded=False):
        st.markdown("**Freecell Deal Structure:**")
        st.code("""
    [C1] [C2] [C3] [C4] [C5] [C6] [C7] [C8]
     01   02   03   04   05   06   07   08
     09   10   11   12   13   14   15   16
     ...
     49   50   51   52
        """, language="text")

    card_input, analysis = render_deck_input(
        st,
        label="Enter your 52 cards (left to right, top to bottom):",
        key="freecell_input",
        sample_seed=1,
        disabled=is_running,
    )

    if analysis.is_valid:
        with st.expander("👀 Board Preview", expanded=False):
            preview_state = FreecellState(list(analysis.cards))
            render_freecell_board(
                st, preview_state.cascades, preview_state.freecells, preview_state.foundations
            )

    goal_mode = st.radio(
        "Goal",
        ["Clear the whole board", "Get one card to the foundation", "Get cards of chosen ranks to the foundation"],
        key="freecell_goal_mode",
        horizontal=True,
        disabled=is_running,
    )
    target_text, ranks_text, rank_count = "", "", None
    one_card = goal_mode == "Get one card to the foundation"
    by_ranks = goal_mode == "Get cards of chosen ranks to the foundation"
    if one_card:
        target_text = st.text_input("Target card (e.g. 4D, TC, 10C)", key="freecell_target_card", disabled=is_running)
    elif by_ranks:
        ranks_col, count_col = st.columns(2)
        with ranks_col:
            ranks_text = st.text_input("Ranks (e.g. 6 or 6, 8, 10, K)", key="freecell_target_ranks", disabled=is_running)
        with count_col:
            rank_count = int(
                st.number_input(
                    "How many of each rank",
                    min_value=1,
                    max_value=4,
                    value=1,
                    step=1,
                    key="freecell_rank_count",
                    disabled=is_running,
                )
            )

    limit_value = st.number_input(
        "Move limit (0 = no limit)",
        min_value=0,
        max_value=500,
        value=0,
        step=1,
        key="freecell_move_limit",
        disabled=is_running,
    )
    move_limit = int(limit_value) or None
    if one_card or by_ranks:
        st.caption("Auto-play is off in this mode. Every action counts as one move (unless atomic move counting is enabled).")
    else:
        st.caption("Moves automatically sent to the foundation are counted as moves when a move limit is set.")

    c1, c2, c3 = st.columns(3)
    with c1:
        minimize_moves = st.checkbox(
            "Minimize Moves",
            value=False,
            key="freecell_minimize_moves",
            disabled=is_running,
            help="Keep searching after the first solution to find the fewest moves.",
        )
    with c2:
        allow_foundation_to_cascade = st.checkbox(
            "Allow Foundation → Cascade",
            value=False,
            key="freecell_allow_f2c",
            disabled=is_running,
            help="Allow pulling cards back down from foundations onto cascades.",
        )
    with c3:
        atomic_move_count = st.checkbox(
            "Atomic Move Counting",
            value=False,
            key="freecell_atomic_moves",
            disabled=is_running,
            help="Count a multi-card supermove of N cards as N moves instead of 1 move.",
        )

    prompt_interval, max_states = render_search_settings(
        st, key_prefix="freecell", default_max_states=DEFAULT_MAX_STATES, disabled=is_running
    )

    if st.button("Solve Freecell", type="primary", disabled=is_running):
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
                    raise ValueError("Ranks: use A, 2-9, 10/T, J, Q or K separated by commas, e.g. 6, 8, K.")
                names = ", ".join("10" if v == 10 else "A23456789TJQK"[v - 1] for v in target_ranks)
                goal_text = f"{rank_count} of each rank ({names}) on the foundations"
            deck_cards = parse_deck(card_input)
        except ValueError as e:
            st.error(f"**Input Error:** {e}")
        except Exception as e:
            st.error("**An unexpected error occurred.**")
            st.exception(e)
        else:
            start_solver_job(
                st,
                job_key=JOB_KEY,
                result_key=RESULT_KEY,
                error_key=ERROR_KEY,
                worker_fn=_run_freecell_search,
                worker_args=(
                    deck_cards,
                    minimize_moves,
                    target_card,
                    move_limit,
                    target_ranks,
                    rank_count,
                    allow_foundation_to_cascade,
                    atomic_move_count,
                    max_states,
                ),
                metadata={
                    "minimize_moves": minimize_moves,
                    "atomic_move_count": atomic_move_count,
                    "goal_text": goal_text,
                    "move_limit": move_limit,
                    "deck_cards": deck_cards,
                },
                prompt_interval_seconds=prompt_interval,
            )

    render_solver_job_area(
        st,
        job_key=JOB_KEY,
        result_key=RESULT_KEY,
        error_key=ERROR_KEY,
        key_prefix="fc",
        build_result_fn=_build_freecell_result,
        render_result_fn=_display_freecell_result,
    )


if __name__ == "__main__":
    main()
