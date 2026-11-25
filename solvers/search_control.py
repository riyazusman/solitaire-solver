from contextlib import contextmanager
from enum import Enum
import sys
from threading import Condition, Lock
from time import monotonic


SEARCH_PROMPT_INTERVAL_SECONDS = 60
# Default 5ms GIL hold by the search thread freezes UI updates
SEARCH_SWITCH_INTERVAL_SECONDS = 0.0001
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 10.0
DEFAULT_PAUSED_ABANDON_TIMEOUT_SECONDS = 15.0

_switch_lock = Lock()
_active_search_count = 0
_original_switch_interval = None


class SearchState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


class SearchStopped(Exception):
    pass


class SearchControl:
    """Thread-safe cooperative pause/resume/stop controller for solver threads.

    Features:
    - Single Condition-guarded state machine (eliminates multi-Event races).
    - Reference-counted sys.setswitchinterval() scoping via active() context manager.
    - Pure active search time accounting (excludes paused idle time).
    - UI heartbeat watchdog to terminate orphaned threads on tab close or page switch.
    - Lightweight fast-path checkpointing and node/progress telemetry.
    """

    def __init__(
        self,
        prompt_interval_seconds=None,
        heartbeat_timeout_seconds=None,
        paused_abandon_timeout_seconds=None,
        watchdog_check_every=256,
    ):
        self._cond = Condition()
        self._state = SearchState.IDLE
        self._stop_flag = False
        self._pause_flag = False
        self._abandoned = False

        self.prompt_interval_seconds = prompt_interval_seconds
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.paused_abandon_timeout_seconds = paused_abandon_timeout_seconds

        step = max(1, int(watchdog_check_every))
        pow2 = 1 << (step.bit_length() - 1)
        self._watchdog_mask = pow2 - 1

        self._nodes = 0
        self._progress_text = ""

        now = monotonic()
        self._last_heartbeat = now
        self._active_seconds = 0.0
        self._interval_active_seconds = 0.0
        self._active_started_at = None

    @contextmanager
    def active(self):
        """Scope GIL switch interval tuning and active-time accounting to a search run."""
        global _active_search_count, _original_switch_interval
        with _switch_lock:
            if _active_search_count == 0:
                _original_switch_interval = sys.getswitchinterval()
                sys.setswitchinterval(SEARCH_SWITCH_INTERVAL_SECONDS)
            _active_search_count += 1

        with self._cond:
            now = monotonic()
            self._last_heartbeat = now
            self._active_started_at = now
            if self._state == SearchState.IDLE:
                self._state = SearchState.RUNNING

        try:
            yield self
        finally:
            with self._cond:
                now = monotonic()
                if self._active_started_at is not None:
                    dt = max(0.0, now - self._active_started_at)
                    self._active_seconds += dt
                    self._interval_active_seconds += dt
                    self._active_started_at = None
                self._state = SearchState.STOPPED
                self._pause_flag = False
                self._cond.notify_all()

            with _switch_lock:
                _active_search_count -= 1
                if _active_search_count == 0 and _original_switch_interval is not None:
                    sys.setswitchinterval(_original_switch_interval)
                    _original_switch_interval = None

    # ---- UI / Controller API ----

    def heartbeat(self):
        """Record that the UI is still actively observing this search."""
        with self._cond:
            self._last_heartbeat = monotonic()

    def request_pause(self):
        with self._cond:
            if self._state == SearchState.RUNNING:
                self._state = SearchState.PAUSING
                self._pause_flag = True
                self._cond.notify_all()

    def request_resume(self):
        with self._cond:
            self._last_heartbeat = monotonic()
            if self._state in (SearchState.PAUSED, SearchState.PAUSING):
                self._pause_flag = False
                self._interval_active_seconds = 0.0
                self._state = SearchState.RUNNING
                self._cond.notify_all()

    def request_stop(self):
        with self._cond:
            self._stop_flag = True
            self._pause_flag = False
            if self._state != SearchState.STOPPED:
                self._state = SearchState.STOPPING
            self._cond.notify_all()

    # ---- Status & Telemetry Properties ----

    @property
    def state(self):
        return self._state

    @property
    def is_paused(self):
        return self._state == SearchState.PAUSED

    @property
    def is_pausing(self):
        return self._state == SearchState.PAUSING

    @property
    def is_stopping(self):
        return self._state == SearchState.STOPPING

    @property
    def stop_requested(self):
        return self._stop_flag

    @property
    def abandoned(self):
        return self._abandoned

    @property
    def nodes_explored(self):
        return self._nodes

    @property
    def progress_text(self):
        return self._progress_text

    def update_progress(self, text: str):
        self._progress_text = text

    @property
    def active_elapsed(self):
        """Total seconds spent actively searching (excluding paused time)."""
        with self._cond:
            if self._active_started_at is not None:
                return self._active_seconds + max(0.0, monotonic() - self._active_started_at)
            return self._active_seconds

    @property
    def interval_elapsed(self):
        """Active search seconds elapsed since the search started or last resumed."""
        with self._cond:
            if self._active_started_at is not None:
                return self._interval_active_seconds + max(0.0, monotonic() - self._active_started_at)
            return self._interval_active_seconds

    # ---- Worker API ----

    def checkpoint(self):
        self._nodes += 1
        if self._stop_flag:
            raise SearchStopped
        if self._pause_flag:
            self._handle_pause()
        if (self._nodes & self._watchdog_mask) == 0:
            self._check_watchdog()

    def _check_watchdog(self):
        now = monotonic()
        should_pause = False
        with self._cond:
            if self._stop_flag:
                raise SearchStopped
            if (
                self.heartbeat_timeout_seconds is not None
                and now - self._last_heartbeat >= self.heartbeat_timeout_seconds
            ):
                self._abandoned = True
                self._stop_flag = True
                self._state = SearchState.STOPPING
                raise SearchStopped

            if self.prompt_interval_seconds is not None and self._active_started_at is not None:
                interval_now = self._interval_active_seconds + max(0.0, now - self._active_started_at)
                if interval_now >= self.prompt_interval_seconds:
                    self._pause_flag = True
                    self._state = SearchState.PAUSING
                    should_pause = True

        if should_pause:
            self._handle_pause()

    def _handle_pause(self):
        with self._cond:
            if self._stop_flag:
                raise SearchStopped
            if not self._pause_flag:
                return

            now = monotonic()
            if self._active_started_at is not None:
                dt = max(0.0, now - self._active_started_at)
                self._active_seconds += dt
                self._interval_active_seconds += dt
                self._active_started_at = None

            self._state = SearchState.PAUSED
            self._cond.notify_all()

            while self._pause_flag and not self._stop_flag:
                self._cond.wait(timeout=0.2)
                if (
                    self.paused_abandon_timeout_seconds is not None
                    and monotonic() - self._last_heartbeat >= self.paused_abandon_timeout_seconds
                ):
                    self._abandoned = True
                    self._stop_flag = True
                    self._state = SearchState.STOPPING
                    break

            if self._stop_flag:
                raise SearchStopped

            self._active_started_at = monotonic()
            self._interval_active_seconds = 0.0
            self._state = SearchState.RUNNING
