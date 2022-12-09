from enum import Enum
from queue import Empty
from typing import Dict, Any, Iterator

from deepdiff import DeepDiff

from nebullvm.core.operations import Operation
from nebullvm.utils.logger import get_logger

log = get_logger()


class AppPhase(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"


class App:
    def __init__(self, root_op: Operation):
        self.root_op = root_op
        self.phase = AppPhase.STARTING
        self._last_state: Dict[str, Any] = {}

    @property
    def state(self):
        return {
            "phase": self.phase,
            **self.root_op.state,
        }

    def stop(self):
        log.info("App is stopping...")
        self.root_op.stop()
        self.phase = AppPhase.STOPPED
        log.info("App stopped")

    def run(self):
        self.phase = AppPhase.RUNNING
        while self.phase is AppPhase.RUNNING:
            self._run_once()

    def visit_tree(self) -> Iterator[Operation]:
        return self.root_op.visit_tree()

    def _state_has_changed(self) -> bool:
        diff = DeepDiff(self._last_state, self.state)
        return len(diff) > 0

    def _check_errors(self):
        try:
            err = self.root_op.error_queue.get(timeout_seconds=0)
            self._err = err
            self.stop()
        except Empty:
            pass

    def _run_once(self):
        # Check if there's any error
        self._check_errors()
        # Avoid running the root operation if the state hasn't changed.
        if self._state_has_changed() is False:
            return
        # Run the root operation
        log.debug("state change detected, running execute")
        self._last_state = self.state
        self.root_op.execute()
