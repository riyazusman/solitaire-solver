from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time

import pytest

from solvers.search_control import (
    SEARCH_SWITCH_INTERVAL_SECONDS,
    SearchControl,
    SearchState,
    SearchStopped,
)
from ui.job_runner import cancel_other_page_jobs


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_switch_interval_restored_on_exit_and_nesting():
    original = sys.getswitchinterval()
    c1 = SearchControl()
    c2 = SearchControl()

    with c1.active():
        assert sys.getswitchinterval() == pytest.approx(SEARCH_SWITCH_INTERVAL_SECONDS)
        with c2.active():
            assert sys.getswitchinterval() == pytest.approx(SEARCH_SWITCH_INTERVAL_SECONDS)
        # Still 1 active search
        assert sys.getswitchinterval() == pytest.approx(SEARCH_SWITCH_INTERVAL_SECONDS)

    assert sys.getswitchinterval() == pytest.approx(original)


def test_switch_interval_restored_after_exception():
    original = sys.getswitchinterval()
    c = SearchControl()

    with pytest.raises(RuntimeError):
        with c.active():
            assert sys.getswitchinterval() == pytest.approx(SEARCH_SWITCH_INTERVAL_SECONDS)
            raise RuntimeError("boom")

    assert sys.getswitchinterval() == pytest.approx(original)
    assert c.state == SearchState.STOPPED


def test_pause_resume_stop_and_active_time_excludes_pause():
    control = SearchControl(watchdog_check_every=1)
    started = threading.Event()
    resumed_after_pause = threading.Event()
    stopped_cleanly = threading.Event()

    def worker():
        with control.active():
            started.set()
            try:
                saw_pause = False
                while True:
                    control.checkpoint()
                    if saw_pause:
                        resumed_after_pause.set()
                    if control.nodes_explored == 5:
                        saw_pause = True
                    time.sleep(0.001)
            except SearchStopped:
                stopped_cleanly.set()

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(timeout=1.0)

    # Let it run briefly, then pause
    time.sleep(0.03)
    control.request_pause()
    assert _wait_until(lambda: control.is_paused, timeout=1.0)
    elapsed_at_pause = control.active_elapsed

    # Sit paused for 0.15s — active_elapsed must NOT grow
    time.sleep(0.15)
    assert control.active_elapsed == pytest.approx(elapsed_at_pause, abs=0.01)

    # Resume and verify it continues
    control.request_resume()
    assert _wait_until(lambda: not control.is_paused, timeout=1.0)
    time.sleep(0.03)

    # Stop cleanly
    control.request_stop()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert stopped_cleanly.is_set()
    assert control.state == SearchState.STOPPED
    # Total active time should be far less than the 0.15s pause + run time
    assert control.active_elapsed < 0.14


def test_heartbeat_watchdog_stops_abandoned_running_worker():
    control = SearchControl(
        heartbeat_timeout_seconds=0.08,
        watchdog_check_every=1,
    )
    stopped = threading.Event()

    def worker():
        with control.active():
            try:
                while True:
                    control.checkpoint()
                    time.sleep(0.002)
            except SearchStopped:
                stopped.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert stopped.is_set()
    assert control.abandoned is True
    assert control.stop_requested is True


def test_heartbeat_watchdog_stops_abandoned_paused_worker():
    control = SearchControl(
        paused_abandon_timeout_seconds=0.1,
        watchdog_check_every=1,
    )
    stopped = threading.Event()

    def worker():
        with control.active():
            try:
                while True:
                    control.checkpoint()
                    time.sleep(0.002)
            except SearchStopped:
                stopped.set()

    t = threading.Thread(target=worker)
    t.start()
    control.request_pause()
    assert _wait_until(lambda: control.is_paused, timeout=1.0)

    # Stop sending heartbeats; paused worker should self-terminate
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert stopped.is_set()
    assert control.abandoned is True


def test_prompt_interval_auto_pauses_worker():
    control = SearchControl(
        prompt_interval_seconds=0.05,
        watchdog_check_every=1,
    )

    def worker():
        with control.active():
            try:
                while True:
                    control.checkpoint()
                    time.sleep(0.002)
            except SearchStopped:
                pass

    t = threading.Thread(target=worker)
    t.start()
    assert _wait_until(lambda: control.is_paused, timeout=1.5)
    control.request_stop()
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_cancel_other_page_jobs_cancels_only_inactive_pages():
    active_ctrl = SearchControl()
    other_ctrl = SearchControl()
    other_exec = ThreadPoolExecutor(max_workers=1)
    active_exec = ThreadPoolExecutor(max_workers=1)

    session_state = {
        "freecell_solver_job": {"search_control": active_ctrl, "executor": active_exec},
        "pyramid_solver_job": {"search_control": other_ctrl, "executor": other_exec},
    }

    cancel_other_page_jobs(session_state, "freecell_solver_job")

    assert "freecell_solver_job" in session_state
    assert "pyramid_solver_job" not in session_state
    assert not active_ctrl.stop_requested
    assert other_ctrl.stop_requested
    active_exec.shutdown(wait=False)
