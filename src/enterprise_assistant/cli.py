"""Command line entry point: `assistant <command>`.

assistant ingest           fetch docs, chunk, embed, load Postgres, build indexes
assistant seed             seed the fictional support desk + CRM, rebuild the benchmark
assistant train            train the reranker (benchmark gold sections held out)
assistant tune             pick the answer-confidence threshold on the dev split
assistant eval             run the benchmark, write reports/, check thresholds
assistant ask "question"   ask one question from the terminal
assistant serve            run the FastAPI app
assistant mcp              run the MCP server (stdio)
assistant all              ingest + seed + train + tune + eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

from .config import get_settings

EVALS = Path("evals")
REPORTS = Path("reports")


def cmd_ingest(args) -> None:
    from .ingest import run_ingest

    run_ingest(get_settings())


def cmd_seed(args) -> None:
    from . import crm, db
    from .corpus import load_chunks
    from .evaluation import benchmark

    s = get_settings()
    chunks = load_chunks(s.artifacts_dir / "chunks.jsonl")
    with db.connect(s.database_url) as conn:
        res = crm.seed(conn, chunks)
        items = benchmark.build(conn, EVALS, res.anchored, res.injected)
    benchmark.write(items, EVALS / "benchmark.jsonl")
    print(
        f"seeded {res.accounts} accounts, {res.contacts} contacts, {res.tickets} tickets; "
        f"benchmark has {len(items)} items"
    )


def cmd_train(args) -> None:
    from .training import train_reranker

    train_reranker(get_settings(), EVALS / "benchmark.jsonl")


def cmd_tune(args) -> None:
    """Pick the answer thresholds (reranker confidence, query coverage) on the dev split.
    Objective: correct answers + correct declines, where answering an out-of-scope question
    costs twice as much as a missed answer (a confident wrong answer is worse than "I don't
    know"). Ties go to the lowest thresholds, so the assistant declines only when it has to."""
    from .evaluation.harness import run_sync

    s = get_settings()
    s.min_answer_confidence = 0.0
    s.min_query_coverage = 0.0
    out = run_sync(s, EVALS / "benchmark.jsonl", split="dev")
    rows = [r for r in out["rows"] if r["type"] in ("docs", "out_of_scope", "mixed")]
    best = (-1.0, 0.0, 0.0)
    for tc in [i / 200 for i in range(0, 61)]:
        for tv in [i / 20 for i in range(0, 19)]:
            ok = 0.0
            for r in rows:
                docs_ok = r["confidence"] >= tc and (r["coverage"] >= tv or r["type"] == "mixed")
                if r["expect"] == "answer":
                    ok += r["correct"] and docs_ok
                elif r["answered"] and docs_ok:
                    ok -= 1.0  # answered something it should have declined
                else:
                    ok += 1.0
            acc = ok / len(rows)
            if acc > best[0] + 1e-9:
                best = (acc, tc, tv)
    acc, tc, tv = best
    print(
        f"min_answer_confidence={tc:.3f} min_query_coverage={tv:.2f} "
        f"(dev objective {acc:.3f} on {len(rows)} items)"
    )
    (s.artifacts_dir / "threshold.json").write_text(
        json.dumps({"min_answer_confidence": tc, "min_query_coverage": tv, "dev_accuracy": acc})
    )


def _apply_threshold(s) -> None:
    path = s.artifacts_dir / "threshold.json"
    env = __import__("os").environ
    if path.exists():
        t = json.loads(path.read_text())
        if "MIN_ANSWER_CONFIDENCE" not in env:
            s.min_answer_confidence = t["min_answer_confidence"]
        if "MIN_QUERY_COVERAGE" not in env:
            s.min_query_coverage = t.get("min_query_coverage", 0.0)


def cmd_eval(args) -> None:
    from .evaluation.harness import run_sync
    from .evaluation.report import check_gates, write_reports

    s = get_settings()
    _apply_threshold(s)
    out = run_sync(s, Path(args.benchmark), split=args.split, limit=args.limit)
    write_reports(out, REPORTS, prefix=args.name)
    print(json.dumps(out["report"], indent=2))
    if args.thresholds:
        gates = yaml.safe_load(Path(args.thresholds).read_text())
        failures = check_gates(out["report"], gates)
        baseline = Path(args.baseline) if args.baseline else None
        if baseline and baseline.exists():
            failures += check_gates(out["report"], gates, json.loads(baseline.read_text()))
        if failures:
            print("\nQUALITY GATE FAILED:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("\nquality gate passed")


def cmd_ask(args) -> None:
    from .runtime import Assistant

    s = get_settings()
    _apply_threshold(s)

    async def go():
        async with Assistant(s) as a:
            r = await a.ask(" ".join(args.question))
        print(r["answer"])
        for c in r["citations"]:
            print(f"  [{c['n']}] {c['title']} > {c['section']}  {c['url']}")
        if r["tool_calls"]:
            print("  tools:", ", ".join(f"{t['name']}({t['args']})" for t in r["tool_calls"]))
        print(
            f"  intent={r['intent']} confidence={r['confidence']} "
            f"latency={sum(r['timings_ms'].values()):.0f}ms flags={r['guardrail_flags']}"
        )

    asyncio.run(go())


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("enterprise_assistant.api:app", host=args.host, port=args.port)


def cmd_mcp(args) -> None:
    from .mcp_server import main

    main()


def cmd_all(args) -> None:
    cmd_ingest(args)
    cmd_seed(args)
    cmd_train(args)
    cmd_tune(args)
    cmd_eval(args)


def main() -> None:
    p = argparse.ArgumentParser(prog="assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "seed", "train", "tune", "mcp"):
        sub.add_parser(name)
    for name in ("eval", "all"):
        e = sub.add_parser(name)
        e.add_argument("--split", choices=["dev", "test"], default=None)
        e.add_argument("--limit", type=int, default=None)
        e.add_argument("--thresholds", default=None, help="YAML quality gates")
        e.add_argument("--baseline", default=None, help="baseline report JSON for regressions")
        e.add_argument("--benchmark", default=str(EVALS / "benchmark.jsonl"))
        e.add_argument("--name", default="eval", help="report file prefix")
    a = sub.add_parser("ask")
    a.add_argument("question", nargs="+")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    try:
        globals()[f"cmd_{args.cmd}"](args)
    except Exception as exc:
        if os.environ.get("GITHUB_ACTIONS"):  # surface the cause as a CI annotation
            msg = f"{type(exc).__name__}: {exc}".replace("\n", "%0A")[:1500]
            print(f"::error title=assistant {args.cmd} failed::{msg}")
        raise


if __name__ == "__main__":
    main()
