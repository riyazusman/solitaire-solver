import streamlit as st

from solvers.models import parse_board_and_stock
from solvers.replay import replay_tripeaks
from solvers.tripeaks import DEFAULT_MAX_STATES, TriPeaksBoard, TriPeaksSolver
from ui.board_views import render_step_playback, render_tripeaks_board
from ui.deck_input import render_deck_input
from ui.job_runner import (
    cancel_other_page_jobs,
    render_search_settings,
    render_solver_job_area,
    start_solver_job,
)

JOB_KEY = "tripeaks_solver_job"
RESULT_KEY = "tripeaks_solver_result"
ERROR_KEY = "tripeaks_solver_error"


def _run_tripeaks_search(board_cards, stock_cards, wrap_enabled, minimize_draws, max_states, search_control):
    solver = TriPeaksSolver(
        board_cards=board_cards,
        stock_cards=stock_cards,
        board_layout=TriPeaksBoard(),
        wrap_enabled=wrap_enabled,
        minimize_draws=minimize_draws,
        search_control=search_control,
        max_states=max_states,
    )
    sequence, max_cleared = solver.solve()
    return (
        sequence,
        max_cleared,
        solver.min_stock_used,
        len(solver.stock_cards),
        solver.state_limit_reached,
        solver.states_visited,
    )


def _build_tripeaks_result(raw_result, search_control, metadata):
    sequence, max_cleared, stock_draws, stock_count, state_limit_reached, states_visited = raw_result
    return {
        "sequence": sequence,
        "max_cleared": max_cleared,
        "elapsed": search_control.active_elapsed,
        "stopped": search_control.stop_requested,
        "state_limit_reached": state_limit_reached,
        "states_visited": states_visited,
        "minimize_draws": metadata["minimize_draws"],
        "stock_draws": stock_draws,
        "stock_count": stock_count,
        "board_cards": metadata["board_cards"],
        "stock_cards": metadata["stock_cards"],
    }


def _display_tripeaks_result(st, result):
    max_cleared = result["max_cleared"]
    stopped = result["stopped"]
    state_limit_reached = result.get("state_limit_reached", False)
    minimize_draws = result["minimize_draws"]
    elapsed = result["elapsed"]

    st.divider()
    if max_cleared == 28:
        if (stopped or state_limit_reached) and minimize_draws:
            reason = "reached the memory safety limit" if state_limit_reached else "stopped"
            st.warning(
                f"🎉 Perfect clear found in {elapsed:.2f}s, but the search {reason} before minimum stock draws were proven."
            )
        else:
            st.success(f"🎉 Perfect Clear Found! (Calculated in {elapsed:.2f}s)")
    elif state_limit_reached:
        st.warning(f"⚠️ Memory safety state limit reached after {elapsed:.2f}s. Showing best partial result found so far.")
    elif stopped:
        st.warning(f"Search stopped after {elapsed:.2f}s. Best result found so far; optimality was not proven.")
    else:
        st.warning(f"⚠️ No perfect clear exists. Best partial clear found in {elapsed:.2f}s.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Cards Cleared", value=f"{max_cleared} / 28")
    with col2:
        st.metric(label="Stock Draws Used", value=f"{result['stock_draws']} / {result['stock_count']}")
    with col3:
        st.metric(label="States Explored", value=f"{result.get('states_visited', 0):,}")

    sequence = result["sequence"]
    if not sequence:
        st.info("No valid moves could be made from the starting layout.")
        return

    text_tab, playback_tab = st.tabs(["📜 Move Sequence", "🎬 Step-by-Step Playback"])
    with text_tab:
        if max_cleared == 28 and minimize_draws and not stopped and not state_limit_reached:
            st.subheader("Move Sequence (fewest stock draws):")
        elif max_cleared == 28 and (stopped or state_limit_reached) and minimize_draws:
            st.subheader("Winning Move Sequence (minimum draws unverified):")
        elif max_cleared == 28:
            st.subheader("Winning Move Sequence:")
        else:
            st.subheader("Best Partial Move Sequence:")
        sequence_text = "".join(f"{step_num}. {step}\n" for step_num, step in enumerate(sequence))
        st.code(sequence_text, language="text")

    with playback_tab:
        board_cards = result["board_cards"]
        stock_cards = result["stock_cards"]
        snapshots = replay_tripeaks(board_cards, stock_cards, sequence)
        render_step_playback(
            st,
            key_prefix="tripeaks",
            snapshots=snapshots,
            render_snapshot_fn=lambda snap: render_tripeaks_board(
                st, board_cards, snap.board_mask, snap.waste, snap.stock_remaining
            ),
        )


def main():
    cancel_other_page_jobs(st.session_state, JOB_KEY)
    job = st.session_state.get(JOB_KEY)
    is_running = job is not None

    st.title("🏔️ Tripeaks Solitaire Solver")
    st.markdown("""
    Find the optimal sequence to clear a Tripeaks board.
    Enter 52 cards separated by commas, spaces, or newlines (e.g., `AS, 2H, 10C, KD...`).
    """)

    with st.expander("📐 Deal Structure Guide", expanded=False):
        st.markdown("**Cards 1-28:** The Board (Top row to bottom row, left to right).")
        st.code("""
      [01]         [02]         [03]          (Row 1: 3 cards)
    [04][05]     [06][07]     [08][09]        (Row 2: 6 cards)
  [10][11][12] [13][14][15] [16][17][18]      (Row 3: 9 cards)
 [19][20][21][22][23][24][25][26][27][28]     (Row 4: 10 cards)
        """, language="text")
        st.markdown(
            "**Cards 29-52:** Card 29 starts face-up on the waste pile; "
            "Cards 30-52 form the 23-card stock pile (left to right)."
        )

    card_input, analysis = render_deck_input(st, key="tripeaks_input", sample_seed=10, disabled=is_running)

    if analysis.is_valid:
        with st.expander("👀 Board Preview", expanded=False):
            cards = list(analysis.cards)
            render_tripeaks_board(st, cards[:28], (1 << 28) - 1, cards[28], 23)

    col1, col2 = st.columns(2)
    with col1:
        wrap_enabled = st.checkbox(
            "Enable King-Ace Wrapping",
            value=True,
            disabled=is_running,
            help="Allow playing an Ace on a King and vice versa.",
        )
    with col2:
        minimize_draws = st.checkbox(
            "Minimize Stock Draws",
            value=False,
            disabled=is_running,
            help="Uses layered 0-1 BFS to prove the absolute minimum number of stock draws required.",
        )

    prompt_interval, max_states = render_search_settings(
        st, key_prefix="tripeaks", default_max_states=DEFAULT_MAX_STATES, disabled=is_running
    )

    if st.button("Solve Tripeaks", type="primary", disabled=is_running):
        try:
            board_cards, stock_cards = parse_board_and_stock(card_input)
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
                worker_fn=_run_tripeaks_search,
                worker_args=(board_cards, stock_cards, wrap_enabled, minimize_draws, max_states),
                metadata={
                    "minimize_draws": minimize_draws,
                    "board_cards": board_cards,
                    "stock_cards": stock_cards,
                },
                prompt_interval_seconds=prompt_interval,
            )

    render_solver_job_area(
        st,
        job_key=JOB_KEY,
        result_key=RESULT_KEY,
        error_key=ERROR_KEY,
        key_prefix="tripeaks",
        build_result_fn=_build_tripeaks_result,
        render_result_fn=_display_tripeaks_result,
    )


if __name__ == "__main__":
    main()
