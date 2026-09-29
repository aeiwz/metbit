"""FastAPI sidecar for the metbit desktop GUI."""
import asyncio
import json
import secrets

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .jobs import Busy, Job, start_job
from .session import Session
from .stages import STAGES, Stage, resolve_schema


def create_app(token: str, stages: list[Stage] = STAGES) -> FastAPI:
    by_id = {s.id: s for s in stages}
    session = Session([s.id for s in stages], {s.id: s.after for s in stages})
    jobs: dict[str, Job] = {}

    def check_token(request: Request) -> None:
        got = request.headers.get("authorization", "")
        if not secrets.compare_digest(got, f"Bearer {token}"):
            raise HTTPException(401, "Invalid token")

    app = FastAPI(dependencies=[Depends(check_token)])
    # ponytail: "*" is safe because every route needs the per-session bearer token;
    # CORS answers preflight before routing, so OPTIONS needs no token.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def stage_or_404(sid: str) -> Stage:
        if sid not in by_id:
            raise HTTPException(404, f"Unknown stage '{sid}'")
        return by_id[sid]

    def result_or_409(sid: str):
        stage_or_404(sid)
        if session.status[sid] not in ("done", "stale"):
            raise HTTPException(409, f"Stage '{sid}' has no result")
        return session.outputs[sid]

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/stages")
    def list_stages():
        return [
            {
                "id": s.id,
                "title": s.title,
                "section": s.section,
                "requires": s.requires,
                "status": session.status[s.id],
                "views": list(s.figures),
                "ppm_views": s.ppm_views,
                "has_table": s.table is not None,
                "blocked": s.blocked(session) if s.blocked else None,
                "schema": resolve_schema(s, session),
                "params": session.params.get(s.id),
            }
            for s in stages
        ]

    @app.post("/stages/{sid}/run")
    def run(sid: str, body: dict = Body(default={})):
        stage = stage_or_404(sid)
        try:
            params = stage.params.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        if stage.requires and session.status[stage.requires] != "done":
            raise HTTPException(409, f"Run '{stage.requires}' first")
        reason = stage.blocked(session) if stage.blocked else None
        if reason:
            raise HTTPException(409, reason)
        try:
            job = start_job(session, stage, params)
        except Busy:
            raise HTTPException(409, "Another job is running")
        jobs[job.id] = job
        return {"job_id": job.id}

    @app.get("/jobs/{job_id}/events")
    async def events(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")

        async def stream():
            while True:
                kind, data = await asyncio.to_thread(job.events.get)
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                if kind in ("done", "error"):
                    jobs.pop(job_id, None)
                    return

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/stages/{sid}/figure")
    def figure(sid: str, view: str):
        out = result_or_409(sid)
        make = by_id[sid].figures.get(view)
        if make is None:
            raise HTTPException(404, f"Unknown view '{view}'")
        fig = make(out)
        fig.update_layout(width=None, height=None, autosize=True)  # renderer sizes the plot
        return Response(fig.to_json(), media_type="application/json")

    @app.get("/stages/{sid}/table")
    def table(sid: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        if stage_or_404(sid).table is None:
            raise HTTPException(404, f"Stage '{sid}' has no table")
        df = by_id[sid].table(result_or_409(sid))
        # double_precision=15 (the max): the default 10 rounds p-values like 3e-11 to 0
        page = json.loads(df.iloc[offset:offset + limit].to_json(orient="split", index=False, double_precision=15))
        return {"columns": [str(c) for c in page["columns"]], "rows": page["data"], "total": len(df)}

    return app
