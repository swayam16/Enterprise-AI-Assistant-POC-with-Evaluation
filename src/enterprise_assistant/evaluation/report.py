"""Write evaluation reports (JSON + Markdown) and check quality gates."""

from __future__ import annotations

import json
from pathlib import Path


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_gates(report: dict, gates: dict, baseline: dict | None = None) -> list[str]:
    failures = []
    if baseline is None:
        for key, floor in (gates.get("min") or {}).items():
            v = _get(report, key)
            if v is None or v < floor:
                failures.append(f"{key} = {v} is below the minimum {floor}")
        for key, ceil in (gates.get("max") or {}).items():
            v = _get(report, key)
            if v is None or v > ceil:
                failures.append(f"{key} = {v} is above the maximum {ceil}")
        return failures
    tol = gates.get("regression_tolerance", 0.02)
    for key in gates.get("no_regression", []):
        new, old = _get(report, key), _get(baseline, key)
        if new is not None and old is not None and new < old - tol:
            failures.append(f"{key} regressed from {old} to {new} (tolerance {tol})")
    return failures


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def markdown(report: dict) -> str:
    a, g, s, r = report["accuracy"], report["groundedness"], report["safety"], report["retrieval"]
    lat, cost = report["latency_ms"], report["cost_per_query"]
    lines = [
        "# Evaluation report",
        "",
        f"Generator: `{report['generator']}` · items: {report['items']} · "
        f"answer thresholds: confidence {report['min_answer_confidence']}, "
        f"coverage {report.get('min_query_coverage', 0)}",
        "",
        "## Answer accuracy",
        "",
        "| Slice | Items | Accuracy |",
        "|---|---:|---:|",
        f"| **All** | {report['items']} | **{_pct(a['overall'])}** |",
        f"| Test split (gated in CI) | | {_pct(a['test_split'])} |",
    ]
    names = {
        "docs": "Documentation questions",
        "ticket": "Ticket lookups (MCP)",
        "crm": "CRM queries (MCP)",
        "mixed": "Ticket + docs (agent, multi-step)",
        "out_of_scope": "Out of scope (should decline)",
        "guardrail": "Guardrail cases (attacks, PII)",
        "injection": "Indirect prompt injection in tickets",
    }
    for t, n in report["counts"].items():
        lines.append(f"| {names.get(t, t)} | {n} | {_pct(a.get(t))} |")
    lines += [
        "",
        "## Groundedness",
        "",
        f"- Answer sentences supported by their cited source: **{_pct(g['mean_sentence_support'])}**",
        f"- Answers where every sentence is supported: {_pct(g['fully_grounded_answers'])}",
        f"- Answers citing the correct documentation page: {_pct(g['cites_correct_page'])}",
        "",
        "## Safety",
        "",
        f"- Out-of-scope questions declined: {_pct(s['out_of_scope_declined'])} "
        f"({s['hallucinated_answers_to_out_of_scope']} answered anyway)",
        f"- Prompt-injection and contact-detail attacks blocked: {_pct(s['attacks_blocked'])}",
        f"- Indirect injections in ticket text resisted: {_pct(s['indirect_injections_resisted'])}",
        f"- Legitimate questions wrongly blocked: {s['false_blocks_on_legit_questions']}",
        f"- Answers containing an email or phone number: {s['answers_with_pii']}",
        "",
        "## Retrieval (gold section rank)",
        "",
        "| Retriever | hit@1 | hit@3 | hit@5 | MRR@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("dense_only", "Dense only (pgvector HNSW)"),
        ("keyword_only", "Keyword only (Postgres full-text)"),
        ("hybrid_rrf", "Hybrid (RRF)"),
        ("hybrid_rrf_rerank", "Hybrid + reranker"),
    ]:
        m = r.get(key) or {}
        if m:
            lines.append(
                f"| {label} | {_pct(m['hit@1'])} | {_pct(m['hit@3'])} | "
                f"{_pct(m['hit@5'])} | {m['mrr@10']:.3f} |"
            )
    lines += (
        [
            "",
            "## Latency (end to end, includes MCP calls over stdio)",
            "",
            f"p50 **{lat['p50']:.0f} ms** · p95 **{lat['p95']:.0f} ms** · mean {lat['mean']:.0f} ms",
            "",
            "| Stage | Mean ms |",
            "|---|---:|",
        ]
        + [f"| {k} | {v} |" for k, v in lat["by_stage_mean"].items()]
        + [
            "",
            "## Cost per query",
            "",
            f"Mean prompt {cost['mean_input_tokens']} tokens, mean answer "
            f"{cost['mean_output_tokens']} tokens. Live LLM cost in this run: "
            f"${cost['live_cost_usd']:.6f}/query.",
            "",
            "| If served by | $/query | $ per 1,000 queries |",
            "|---|---:|---:|",
        ]
        + [f"| {m} | ${v:.5f} | ${v * 1000:.2f} |" for m, v in cost["projected_usd"].items()]
    )
    return "\n".join(lines) + "\n"


def write_reports(out: dict, directory: Path, prefix: str = "eval") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{prefix}_report.json").write_text(json.dumps(out["report"], indent=2))
    (directory / f"{prefix}_report.md").write_text(markdown(out["report"]))
    with (directory / f"{prefix}_rows.jsonl").open("w") as f:
        for row in out["rows"]:
            f.write(json.dumps(row, default=str) + "\n")
