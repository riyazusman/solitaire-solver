from concurrent.futures import ThreadPoolExecutor

from solvers.search_control import (
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_PAUSED_ABANDON_TIMEOUT_SECONDS,
    SEARCH_PROMPT_INTERVAL_SECONDS,
    SearchControl,
)

ALL_JOB_SESSION_KEYS = (
    "tripeaks_solver_job",
    "pyramid_solver_job",
    "freecell_solver_job",
)


def cancel_other_page_jobs(session_state, active_job_key: str):
    """Stop and clean up any running solver jobs belonging to inactive pages."""
    for key in ALL_JOB_SESSION_KEYS:
        if key == active_job_key:
            continue
        job = session_state.pop(key, None)
        if job is not None:
            control = job.get("search_control")
            if control is not None:
                control.request_stop()
            executor = job.get("executor")
            if executor is not None:
                executor.shutdown(wait=False)


def render_search_settings(
    st,
    *,
    key_prefix: str,
    default_max_states: int = 2_000_000,
    disabled: bool = False,
) -> tuple[int, int]:
    """Render configurable pause interval and memory-safety state limit inside an expander."""
    with st.expander("⚙️ Search Settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            prompt_interval = st.selectbox(
                "Pause prompt interval",
                options=[30, 60, 120, 300],
                index=1,
                format_func=lambda s: f"{s} seconds",
                key=f"{key_prefix}_prompt_interval",
                disabled=disabled,
            )
        with c2:
            state_options = [500_000, 1_000_000, 2_000_000, 5_000_000]
            default_idx = state_options.index(default_max_states) if default_max_states in state_options else 2
            max_states = st.selectbox(
                "Memory safety cap (max states)",
                options=state_options,
                index=default_idx,
                format_func=lambda n: f"{n:,} states",
                key=f"{key_prefix}_max_states",
                disabled=disabled,
            )
    return int(prompt_interval), int(max_states)


def _run_with_active_control(worker_fn, search_control, worker_args):
    with search_control.active():
        return worker_fn(*worker_args, search_control)


def start_solver_job(
    st,
    *,
    job_key: str,
    result_key: str,
    error_key: str,
    worker_fn,
    worker_args: tuple,
    metadata: dict,
    prompt_interval_seconds: int = SEARCH_PROMPT_INTERVAL_SECONDS,
):
    """Submit a solver run to a background thread and trigger an immediate rerun."""
    st.session_state.pop(result_key, None)
    st.session_state.pop(error_key, None)

    search_control = SearchControl(
        prompt_interval_seconds=prompt_interval_seconds,
        heartbeat_timeout_seconds=DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        paused_abandon_timeout_seconds=DEFAULT_PAUSED_ABANDON_TIMEOUT_SECONDS,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        _run_with_active_control,
        worker_fn,
        search_control,
        worker_args,
    )
    st.session_state[job_key] = {
        "future": future,
        "executor": executor,
        "search_control": search_control,
        "prompt_interval_seconds": prompt_interval_seconds,
        "metadata": metadata,
    }
    st.rerun()


def render_solver_job_area(
    st,
    *,
    job_key: str,
    result_key: str,
    error_key: str,
    key_prefix: str,
    build_result_fn,
    render_result_fn,
):
    """Render either the live polling fragment for an active job, an error, or the final result."""
    job = st.session_state.get(job_key)
    if job is not None:
        @st.fragment(run_every="1s")
        def render_search_status():
            active_job = st.session_state.get(job_key)
            if active_job is None:
                return

            future = active_job["future"]
            search_control = active_job["search_control"]
            prompt_interval = active_job["prompt_interval_seconds"]
            search_control.heartbeat()

            if future.done():
                active_job["executor"].shutdown(wait=False)
                st.session_state.pop(job_key, None)
                try:
                    raw_result = future.result()
                except Exception as exc:
                    st.session_state[error_key] = exc
                    st.rerun()
                    return

                st.session_state[result_key] = build_result_fn(
                    raw_result,
                    search_control,
                    active_job["metadata"],
                )
                st.rerun()
                return

            if search_control.is_stopping:
                st.info("Stopping search and collecting the best result found so far...")
                return

            if search_control.is_paused:
                elapsed_active = int(search_control.active_elapsed)
                progress_info = (
                    f" ({search_control.progress_text})" if search_control.progress_text else ""
                )
                st.info(
                    f"Search paused after {elapsed_active}s of active search{progress_info}. "
                    "Continue searching or stop and show the best result so far."
                )
                continue_col, stop_col = st.columns(2)
                with continue_col:
                    if st.button("Continue searching", key=f"{key_prefix}_continue_search"):
                        search_control.request_resume()
                        st.rerun(scope="fragment")
                with stop_col:
                    if st.button("Stop and show best result", key=f"{key_prefix}_stop_search"):
                        search_control.request_stop()
                        st.rerun(scope="fragment")
                return

            if (
                not search_control.is_pausing
                and search_control.interval_elapsed >= prompt_interval
            ):
                search_control.request_pause()

            elapsed_total = int(search_control.active_elapsed)
            label = (
                "Pausing at the next search checkpoint"
                if search_control.is_pausing
                else "Searching for a solution"
            )
            progress_suffix = (
                f" | {search_control.progress_text}" if search_control.progress_text else ""
            )
            st.status(
                f"{label}... {elapsed_total}s active ({search_control.nodes_explored:,} checks){progress_suffix}",
                state="running",
                expanded=False,
            )
            if st.button("Stop and show best result", key=f"{key_prefix}_stop_running"):
                search_control.request_stop()
                st.rerun(scope="fragment")

        render_search_status()
    elif error_key in st.session_state:
        st.error("**An unexpected error occurred.**")
        st.exception(st.session_state[error_key])
    elif result_key in st.session_state:
        render_result_fn(st, st.session_state[result_key])
