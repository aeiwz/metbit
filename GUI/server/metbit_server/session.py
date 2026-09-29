"""In-memory pipeline state for one GUI session."""
import threading
from typing import Any, Optional


class Session:
    def __init__(self, stage_ids: list[str], after: Optional[dict[str, list[str]]] = None) -> None:
        self.order = list(stage_ids)
        self.after = after or {}  # stage id -> ids whose re-run makes it stale
        self.status: dict[str, str] = {sid: "empty" for sid in self.order}
        self.params: dict[str, dict] = {}
        self.outputs: dict[str, Any] = {}
        self.lock = threading.Lock()  # one job at a time

    def dependents(self, sid: str) -> list[str]:
        """Every stage that depends on `sid`, directly or transitively."""
        found: list[str] = []
        todo = [sid]
        while todo:
            cur = todo.pop()
            for other in self.order:
                if cur in self.after.get(other, ()) and other not in found:
                    found.append(other)
                    todo.append(other)
        return found

    def start(self, sid: str) -> None:
        self.status[sid] = "running"

    def complete(self, sid: str, params: dict, output: Any) -> None:
        self.params[sid] = params
        self.outputs[sid] = output
        self.status[sid] = "done"
        for dep in self.dependents(sid):
            if self.status[dep] == "done":
                self.status[dep] = "stale"

    def fail(self, sid: str) -> None:
        self.status[sid] = "failed"

    def output(self, sid: str) -> Any:
        if sid not in self.outputs:
            raise KeyError(sid)
        return self.outputs[sid]
