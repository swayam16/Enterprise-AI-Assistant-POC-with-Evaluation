"""FastAPI service: the assistant behind a small JSON API and a demo page.

POST /v1/ask        {"question": "..."} -> answer, citations, tool calls, guardrail flags,
                    per-stage timings, tokens and cost
GET  /v1/metrics    running counters since start (requests, blocks, declines, latency, cost)
GET  /v1/report     the latest evaluation report, if one has been generated
GET  /healthz       liveness + dependency check
GET  /              demo UI
"""

from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .cli import _apply_threshold
from .config import get_settings
from .runtime import Assistant


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class Metrics:
    def __init__(self) -> None:
        self.requests = self.blocked = self.declined = 0
        self.latencies: list[float] = []
        self.cost = 0.0
        self.projected_cost = 0.0

    def record(self, resp: dict, latency_ms: float) -> None:
        self.requests += 1
        self.blocked += resp["blocked"]
        self.declined += (not resp["answered"]) and not resp["blocked"]
        self.latencies = (self.latencies + [latency_ms])[-5000:]
        self.cost += resp["usage"]["cost_usd"] or 0.0
        self.projected_cost += resp["usage"]["projected_cost_usd"] or 0.0

    def snapshot(self) -> dict:
        lat = sorted(self.latencies)
        return {
            "requests": self.requests,
            "blocked": self.blocked,
            "declined": self.declined,
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None,
            "llm_cost_usd_total": round(self.cost, 6),
            "projected_cost_usd_total": round(self.projected_cost, 6),
        }


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _apply_threshold(settings)
    async with Assistant(settings) as assistant:
        state["assistant"] = assistant
        state["metrics"] = Metrics()
        yield
    state.clear()


app = FastAPI(title="Enterprise AI Assistant POC", version="0.1.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("ASSISTANT_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def public(resp: dict) -> dict:
    return {k: v for k, v in resp.items() if not k.startswith("_")}


@app.post("/v1/ask", dependencies=[Depends(require_key)])
async def ask(req: AskRequest) -> dict:
    t0 = time.perf_counter()
    resp = await state["assistant"].ask(req.question)
    latency = (time.perf_counter() - t0) * 1000
    state["metrics"].record(resp, latency)
    return {"request_id": str(uuid.uuid4()), "latency_ms": round(latency, 1), **public(resp)}


@app.get("/v1/metrics", dependencies=[Depends(require_key)])
async def metrics() -> dict:
    return state["metrics"].snapshot()


@app.get("/v1/report")
async def report() -> dict:
    path = Path("reports/eval_report.json")
    if not path.exists():
        raise HTTPException(404, "No evaluation report yet. Run `assistant eval`.")
    return json.loads(path.read_text())


@app.get("/healthz")
async def healthz() -> dict:
    a = state.get("assistant")
    if a is None:
        raise HTTPException(503, "starting")
    with a.pool.connection() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    tools = [t["name"] for t in await a.tools.list()]
    return {
        "status": "ok",
        "chunks": chunks,
        "mcp_tools": tools,
        "llm_provider": a.settings.llm_provider,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return resources.files("enterprise_assistant").joinpath("static/index.html").read_text()
