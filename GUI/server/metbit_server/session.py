"""In-memory pipeline state for one GUI session."""
import threading
from typing import Any


class Session:
    def __init__(self, stage_ids: list[str]) -> None:
        self.order = list(stage_ids)
        self.status: dict[str, str] = {sid: "empty" for sid in self.order}
        self.params: dict[str, dict] = {}
        self.outputs: dict[str, Any] = {}
        self.lock = threading.Lock()  # one job at a time

    def start(self, sid: str) -> None:
        self.status[sid] = "running"

    def complete(self, sid: str, params: dict, output: Any) -> None:
        self.params[sid] = params
        self.outputs[sid] = output
        self.status[sid] = "done"
        # ponytail: stages are linear, so "downstream" means "later in order"
        for later in self.order[self.order.index(sid) + 1:]:
            if self.status[later] == "done":
                self.status[later] = "stale"

    def fail(self, sid: str) -> None:
        self.status[sid] = "failed"

    def output(self, sid: str) -> Any:
        if sid not in self.outputs:
            raise KeyError(sid)
        return self.outputs[sid]
