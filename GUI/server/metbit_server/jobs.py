"""Run one stage at a time on a worker thread and queue its events for SSE."""
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel


class Busy(Exception):
    """Another job holds the session lock."""


@dataclass
class Job:
    id: str
    events: "queue.Queue[tuple[str, dict]]" = field(default_factory=queue.Queue)


def start_job(session, stage, params: BaseModel) -> Job:
    if not session.lock.acquire(blocking=False):
        raise Busy()
    job = Job(uuid.uuid4().hex)
    session.start(stage.id)

    def progress(pct: float) -> None:
        job.events.put(("progress", {"pct": round(float(pct), 3)}))

    def work() -> None:
        try:
            out = stage.run(params, session, progress)
            session.complete(stage.id, params.model_dump(), out)
            event = ("done", {"stage": stage.id})
        except Exception as e:
            session.fail(stage.id)
            event = ("error", {"message": str(e), "traceback": traceback.format_exc()})
        # release before announcing, so a client reacting to "done" can run again at once
        session.lock.release()
        job.events.put(event)

    threading.Thread(target=work, daemon=True).start()
    return job
