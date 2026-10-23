import sys
from threading import Event


SEARCH_PROMPT_INTERVAL_SECONDS = 60
# Default 5ms GIL hold by the search thread freezes UI updates
SEARCH_SWITCH_INTERVAL_SECONDS = 0.0001


class SearchStopped(Exception):
    pass


class SearchControl:
    def __init__(self):
        sys.setswitchinterval(SEARCH_SWITCH_INTERVAL_SECONDS)
        self.stop_requested = Event()
        self.pause_requested = Event()
        self.paused = Event()
        self.resume_requested = Event()

    def checkpoint(self):
        if self.stop_requested.is_set():
            raise SearchStopped

        if self.pause_requested.is_set():
            self.paused.set()
            while not self.resume_requested.wait(timeout=0.1):
                if self.stop_requested.is_set():
                    break
            self.paused.clear()
            self.pause_requested.clear()
            self.resume_requested.clear()

        if self.stop_requested.is_set():
            raise SearchStopped
