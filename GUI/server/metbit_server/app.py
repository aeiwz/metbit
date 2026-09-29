"""FastAPI sidecar for the metbit desktop GUI."""
import asyncio
import json
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError, field_validator

from .annotations import ANNOTATED_VIEWS, AnnotationIn, new_annotation, overlay, parse_annotation_file
from .jobs import Busy, Job, start_job
from .project import ProjectError, load_project, save_project
from .session import Session
from .stages import STAGES, Stage, resolve_schema
from .stages.base import option_values, working
from .stages.style import PlotStyle, StyleOverride, merge, override_dict, render, with_groups


class PathBody(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _usable(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("Use an absolute path")
        if not p.parent.is_dir():
            raise ValueError(f"Folder does not exist: {p.parent}")
        return v


def create_app(token: str, stages: list[Stage] = STAGES, autosave: Optional[str] = None) -> FastAPI:
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

    @contextmanager
    def exclusive():
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "Another job is running")
        try:
            yield
        finally:
            session.lock.release()

    def global_style() -> PlotStyle:
        return with_groups(PlotStyle.model_validate(session.style or {}), option_values(session, "groups") or [])

    def effective_style(sid: str) -> PlotStyle:
        return merge(global_style(), StyleOverride.model_validate(session.stage_styles.get(sid, {})))

    def stage_style_body(sid: str) -> dict:
        return {"override": session.stage_styles.get(sid, {}), "effective": effective_style(sid).model_dump()}

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
                "pre_views": list(s.pre_views),
                "ppm_views": s.ppm_views,
                "has_table": s.table is not None,
                "exportable": (s.export or s.table) is not None,
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
            job = start_job(session, stage, params, on_done=autosave_now if autosave else None)
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
        stage = stage_or_404(sid)
        style = effective_style(sid)
        if view in stage.pre_views:
            try:
                data = working(session)  # pre-run views draw from the working matrix, no result needed
            except ValueError as e:
                raise HTTPException(409, str(e))
            fig = render(stage.pre_views[view], data, style)
        else:
            out = result_or_409(sid)
            make = stage.figures.get(view)
            if make is None:
                raise HTTPException(404, f"Unknown view '{view}'")
            fig = render(make, out, style)
        if view in ANNOTATED_VIEWS and session.annotations:
            overlay(fig, session.annotations, style.font_size)
        fig.update_layout(width=None, height=None, autosize=True)  # renderer sizes the plot
        return Response(fig.to_json(), media_type="application/json")

    @app.get("/style")
    def get_style():
        return global_style().model_dump()

    @app.put("/style")
    def put_style(body: dict = Body(...)):
        try:
            style = PlotStyle.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.style = style.model_dump()
        return global_style().model_dump()

    @app.get("/stages/{sid}/style")
    def get_stage_style(sid: str):
        stage_or_404(sid)
        return stage_style_body(sid)

    @app.put("/stages/{sid}/style")
    def put_stage_style(sid: str, body: dict = Body(...)):
        stage_or_404(sid)
        try:
            override = StyleOverride.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.stage_styles[sid] = override_dict(override)
        if not session.stage_styles[sid]:
            del session.stage_styles[sid]
        return stage_style_body(sid)

    @app.get("/stages/{sid}/table")
    def table(sid: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        if stage_or_404(sid).table is None:
            raise HTTPException(404, f"Stage '{sid}' has no table")
        df = by_id[sid].table(result_or_409(sid))
        # double_precision=15 (the max): the default 10 rounds p-values like 3e-11 to 0
        page = json.loads(df.iloc[offset:offset + limit].to_json(orient="split", index=False, double_precision=15))
        return {"columns": [str(c) for c in page["columns"]], "rows": page["data"], "total": len(df)}

    @app.post("/stages/{sid}/export")
    def export(sid: str, body: PathBody):
        stage = stage_or_404(sid)
        make = stage.export or stage.table
        if make is None:
            raise HTTPException(404, f"Stage '{sid}' has nothing to export as CSV")
        make(result_or_409(sid)).to_csv(body.path, index=False)
        return {"path": body.path}

    @app.get("/annotations")
    def list_annotations():
        return session.annotations

    @app.post("/annotations")
    def add_annotation(body: dict = Body(...)):
        try:
            item = new_annotation(AnnotationIn.model_validate(body))
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        session.annotations.append(item)
        return item

    @app.delete("/annotations/{aid}")
    def delete_annotation(aid: str):
        for i, a in enumerate(session.annotations):
            if a["id"] == aid:
                del session.annotations[i]
                return {"deleted": aid}
        raise HTTPException(404, "Unknown annotation")

    @app.post("/annotations/import")
    def import_annotations(body: PathBody):
        try:
            items = parse_annotation_file(Path(body.path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise HTTPException(422, f"Cannot read annotations: {e}")
        session.annotations[:] = items  # in place: the Annotate result shares this list
        return session.annotations

    @app.post("/annotations/export")
    def export_annotations(body: PathBody):
        Path(body.path).write_text(json.dumps(session.annotations, indent=2), encoding="utf-8")
        return {"path": body.path}

    def autosave_now() -> None:
        save_project(session, autosave)

    @app.post("/session/save")
    def save(body: PathBody):
        with exclusive():
            save_project(session, body.path)
        return {"path": body.path}

    @app.post("/session/load")
    def load(body: PathBody):
        with exclusive():
            try:
                load_project(session, stages, body.path)
            except ProjectError as e:
                raise HTTPException(422, str(e))
        return {"path": body.path}

    @app.post("/session/restore")
    def restore():
        if not autosave or not Path(autosave).is_file():
            return {"restored": False}
        with exclusive():
            try:
                load_project(session, stages, autosave)
            except ProjectError:
                return {"restored": False}
        return {"restored": True}

    return app
