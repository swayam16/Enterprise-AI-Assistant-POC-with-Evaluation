"""The assistant as a LangGraph state machine.

    input_guard ─┬─> plan ──> act (MCP tools) ─┬─> retrieve ──> generate ──> output_guard ──> finalize
                 │                             └──────────────> generate
                 ├─> llm_agent (Claude/GPT tool-use loop over MCP tools + search_docs) ──> output_guard
                 └─> finalize (blocked)

Offline mode plans with deterministic rules (ticket IDs, account names, CRM filters), so the
benchmark is reproducible without an API key. With LLM_PROVIDER=anthropic the model itself
chooses tools in a tool-use loop.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import PRICES_PER_MTOK, Settings
from .guardrails import check_input, check_output, content_tokens, redact, sanitize_tool_output
from .llm import (
    NOT_FOUND,
    SYSTEM_PROMPT,
    Generation,
    OfflineGenerator,
    build_user_message,
    cost_usd,
    estimate_tokens,
    unknown_terms,
    weighted_coverage,
)
from .search import Candidate, hybrid_search

DECLINE_MSG = (
    "I couldn't find this in the GitLab documentation or the support systems I can "
    "access, so I won't guess. Try rephrasing, or route it to a support engineer."
)
BLOCK_MSGS = {
    "prompt_injection": "I can't help with that. I answer questions about GitLab using the "
    "product documentation and our support systems.",
    "contact_details": "I can't share customer contact details. Authorized staff can find them "
    "in the CRM.",
    "empty": "Please enter a question.",
    "too_long": "That question is too long. Please shorten it to under 2,000 characters.",
}
TICKET_RE = re.compile(r"\bSUP-\d{4}\b", re.I)
FIX_RE = re.compile(
    r"\b(fix|resolve|solve|solution|recommend|workaround|documentation|docs|"
    r"how (should|do|can) we|how to)\b",
    re.I,
)
CRM_LIST_RE = re.compile(r"\b(which|what|list|show)\b[^?]*\b(accounts|customers)\b", re.I)
MONTHS = "january february march april may june july august september october november december"


class State(TypedDict, total=False):
    question: str
    clean_question: str
    blocked: bool
    block_reason: str
    flags: list[str]
    intent: str
    plan: list[dict]
    tool_results: list[dict]
    need_docs: bool
    search_query: str
    candidates: list[Candidate]
    contexts: list[Candidate]
    confidence: float
    coverage: float
    unknown_terms: list[str]
    generation: Generation
    answer: str
    answered: bool
    timings: dict[str, float]
    llm_calls: list[dict]


@dataclass
class Runtime:
    settings: Settings
    embedder: Any
    reranker: Any
    pool: Any  # psycopg_pool.ConnectionPool
    tools: Any  # ToolBox (MCP client wrapper)
    generator: Any = field(default_factory=OfflineGenerator)
    llm_client: Any = None  # anthropic client, for tool-use mode
    account_names: list[str] = field(default_factory=list)

    @property
    def account_re(self) -> re.Pattern | None:
        if not self.account_names:
            return None
        names = sorted(self.account_names, key=len, reverse=True)
        return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)


def _timed(state: State, name: str, t0: float) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[name] = round((time.perf_counter() - t0) * 1000, 2)
    return timings


def tool_evidence(result: Any) -> str:
    """Flatten tool JSON into words (keys de-snake-cased, dates and money also in display
    form) so the grounding check compares like with like."""
    out: list[str] = []

    def walk(v: Any, key: str = "") -> None:
        if isinstance(v, dict):
            if key:
                out.append(key.replace("_", " "))
            for k, x in v.items():
                walk(x, k)
            return
        if isinstance(v, list):
            for x in v:
                walk(x, key)
            return
        out.append(key.replace("_", " "))
        if isinstance(v, bool) or v is None:
            out.append(str(v))
        elif isinstance(v, int):
            out.extend([str(v), f"{v:,}", f"${v:,}"])
        else:
            s = str(v)
            out.append(s)
            if re.match(r"\d{4}-\d{2}-\d{2}T", s):
                out.append(s[:10])

    walk(result)
    return " ".join(out)


# --- Rule-based planner (offline mode) ----------------------------------------------------------


def _parse_date(text: str) -> str | None:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    m = re.search(rf"\b({MONTHS.replace(' ', '|')})\s+(\d{{1,2}}),?\s+(\d{{4}})", text, re.I)
    if m:
        month = MONTHS.split().index(m.group(1).lower()) + 1
        return date(int(m.group(3)), month, int(m.group(2))).isoformat()
    return None


def crm_filters(q: str) -> dict[str, Any]:
    f: dict[str, Any] = {}
    if m := re.search(r"\b(free|premium|ultimate)\b", q, re.I):
        f["plan"] = m.group(1).capitalize()
    if m := re.search(r"\b(NA|EMEA|APAC|LATAM)\b", q):
        f["region"] = m.group(1)
    before = re.search(r"\b(?:on or before|before|by|until|no later than)\s+([^?]+)", q, re.I)
    if before and (d := _parse_date(before.group(1))):
        f["renewal_before"] = d
    after = re.search(r"\b(?:after|from)\s+([^?]+)", q, re.I)
    if after and (d := _parse_date(after.group(1))):
        f["renewal_after"] = d
    if m := re.search(r"health score (?:of )?(\d+) or (?:lower|less|below)", q, re.I):
        f["max_health_score"] = int(m.group(1))
    elif m := re.search(r"health score (?:below|under|less than) (\d+)", q, re.I):
        f["max_health_score"] = int(m.group(1)) - 1
    if m := re.search(r"(?:at least|over|more than|above)\s+\$?([\d,]+)\s*(k\b)?", q, re.I):
        n = int(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        f["min_arr_usd"] = n
    if m := re.search(r"\b(?:unsolved|open)\s+(P[1-4])\b", q, re.I):
        f["has_unsolved_priority"] = m.group(1).upper()
    return f


def plan_rules(q: str, account_re: re.Pattern | None) -> tuple[str, list[dict], bool]:
    """Return (intent, tool calls, need_docs)."""
    tickets = [t.upper() for t in TICKET_RE.findall(q)]
    accounts = [m.group(1) for m in account_re.finditer(q)] if account_re else []
    if tickets:
        calls = [{"name": "get_ticket", "args": {"ticket_id": t}} for t in dict.fromkeys(tickets)]
        if FIX_RE.search(q):
            return "ticket_docs", calls, True
        return "ticket", calls, False
    if accounts:
        acc = accounts[0]
        if re.search(r"\b(list|show)\b.*\btickets\b|\bwhich tickets\b", q, re.I):
            status = "open" if re.search(r"\bopen\b", q, re.I) else None
            return (
                "tickets",
                [{"name": "search_tickets", "args": {"account": acc, "status": status}}],
                False,
            )
        return "account", [{"name": "get_account", "args": {"name_or_id": acc}}], False
    if CRM_LIST_RE.search(q):
        filters = crm_filters(q)
        if filters:
            return "crm_query", [{"name": "query_accounts", "args": filters}], False
    return "docs", [], True


# --- Graph nodes ----------------------------------------------------------------------------------


def build_graph(rt: Runtime):
    llm_mode = rt.settings.llm_provider in ("anthropic",) and rt.llm_client is not None

    def input_guard(state: State) -> State:
        t0 = time.perf_counter()
        g = check_input(state["question"], rt.account_re)
        return {
            "clean_question": g.text,
            "blocked": g.blocked,
            "block_reason": g.reason,
            "flags": g.flags,
            "timings": _timed(state, "input_guard", t0),
        }

    def plan(state: State) -> State:
        t0 = time.perf_counter()
        intent, calls, need_docs = plan_rules(state["clean_question"], rt.account_re)
        return {
            "intent": intent,
            "plan": calls,
            "need_docs": need_docs,
            "timings": _timed(state, "plan", t0),
        }

    async def act(state: State) -> State:
        t0 = time.perf_counter()
        results, flags = [], list(state.get("flags", []))
        for call in state.get("plan", []):
            args = {k: v for k, v in call["args"].items() if v is not None}
            result = await rt.tools.call(call["name"], args)
            result, flagged = _sanitize_result(result)
            if flagged:
                flags.append("tool_injection_removed")
            results.append({"name": call["name"], "args": args, "result": result})
        query = state["clean_question"]
        if state.get("intent") == "ticket_docs":
            # Search the docs with what the customer actually reported.
            found = [r["result"] for r in results if "subject" in r["result"]]
            if found:
                query = " ".join(f"{r['subject']}. {r['description']}" for r in found)
            else:
                return {
                    "tool_results": results,
                    "need_docs": False,
                    "flags": flags,
                    "timings": _timed(state, "tools", t0),
                }
        return {
            "tool_results": results,
            "search_query": query,
            "flags": flags,
            "timings": _timed(state, "tools", t0),
        }

    def retrieve(state: State) -> State:
        t0 = time.perf_counter()
        query = state.get("search_query") or state["clean_question"]
        qvec = rt.embedder.embed([query])[0]
        with rt.pool.connection() as conn:
            cands = hybrid_search(conn, query, qvec, k=rt.settings.candidates_k)
        timings = _timed(state, "retrieve", t0)
        t1 = time.perf_counter()
        ranked = rt.reranker.score(query, cands)
        contexts = ranked[: rt.settings.rerank_k]
        conf = contexts[0].score if contexts else 0.0
        seen: set[str] = set()
        for c in contexts[:3]:
            seen |= content_tokens(f"{c.title} {c.section} {c.text}")
        q_tokens = content_tokens(re.sub(r"\[REDACTED_[A-Z]+\]", " ", query))
        coverage = weighted_coverage(q_tokens, seen)
        timings["rerank"] = round((time.perf_counter() - t1) * 1000, 2)
        return {
            "candidates": ranked,
            "contexts": contexts,
            "confidence": conf,
            "coverage": coverage,
            "unknown_terms": unknown_terms(q_tokens),
            "timings": timings,
        }

    def generate(state: State) -> State:
        t0 = time.perf_counter()
        need_docs = state.get("need_docs", False)
        contexts = state.get("contexts", [])
        confident = state.get("confidence", 0.0) >= rt.settings.min_answer_confidence
        if state.get("intent") == "docs":  # a ticket's narrative isn't a question; skip there
            confident = (
                confident
                and not state.get("unknown_terms")
                and state.get("coverage", 0.0) >= rt.settings.min_query_coverage
            )
        if need_docs and not confident:
            contexts = []
        q = None
        if state.get("intent") == "ticket_docs" and isinstance(rt.generator, OfflineGenerator):
            q = "How do I fix: " + state.get("search_query", "")  # extract the documented fix
        gen = rt.generator.generate(
            q or state["clean_question"], contexts, state.get("tool_results", []), need_docs
        )
        if state.get("intent") == "ticket_docs" and not gen.found:
            # Still useful: the ticket summary, plus an honest "no documented fix".
            ticket_part = (
                OfflineGenerator()
                .generate(state["clean_question"], [], state.get("tool_results", []), False)
                .text
            )
            gen.text = (ticket_part + " I couldn't find a documented fix for this issue.").strip()
            gen.found = bool(ticket_part)
        answered = gen.found and bool(gen.text)
        return {
            "generation": gen,
            "answer": gen.text if answered else DECLINE_MSG,
            "answered": answered,
            "contexts": contexts if need_docs else [],
            "llm_calls": [_usage(gen)],
            "timings": _timed(state, "generate", t0),
        }

    async def llm_agent(state: State) -> State:
        t0 = time.perf_counter()
        out = await run_tool_use_loop(rt, state["clean_question"])
        out["timings"] = _timed(state, "llm_agent", t0)
        out["flags"] = list(state.get("flags", [])) + out.get("flags", [])
        return out

    def output_guard(state: State) -> State:
        t0 = time.perf_counter()
        if not state.get("answered"):
            return {"timings": _timed(state, "output_guard", t0)}
        evidence = [f"{c.title} {c.section} {c.text}" for c in state.get("contexts", [])]
        evidence += [tool_evidence(t["result"]) for t in state.get("tool_results", [])]
        g = check_output(state["answer"], evidence)
        flags = list(state.get("flags", [])) + g.flags
        if g.blocked:
            return {
                "answer": DECLINE_MSG,
                "answered": False,
                "flags": flags,
                "timings": _timed(state, "output_guard", t0),
            }
        return {"answer": g.text, "flags": flags, "timings": _timed(state, "output_guard", t0)}

    def finalize(state: State) -> State:
        if state.get("blocked"):
            return {
                "answer": BLOCK_MSGS.get(state.get("block_reason", ""), DECLINE_MSG),
                "answered": False,
                "intent": "blocked",
            }
        return {}

    g = StateGraph(State)
    g.add_node("input_guard", input_guard)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("llm_agent", llm_agent)
    g.add_node("output_guard", output_guard)
    g.add_node("finalize", finalize)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges(
        "input_guard",
        lambda s: "finalize" if s.get("blocked") else ("llm_agent" if llm_mode else "plan"),
        ["finalize", "llm_agent", "plan"],
    )
    g.add_edge("plan", "act")
    g.add_conditional_edges(
        "act", lambda s: "retrieve" if s.get("need_docs") else "generate", ["retrieve", "generate"]
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "output_guard")
    g.add_edge("llm_agent", "output_guard")
    g.add_edge("output_guard", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def _sanitize_result(result: dict) -> tuple[dict, bool]:
    flagged = False
    clean = {}
    for k, v in result.items():
        if isinstance(v, str):
            v, f = sanitize_tool_output(v)
            v, _ = redact(v) if "@" in v and "***" not in v else (v, None)
            flagged |= f
        clean[k] = v
    return clean, flagged


def _usage(gen: Generation) -> dict:
    return {
        "model": gen.model,
        "input_tokens": gen.input_tokens,
        "output_tokens": gen.output_tokens,
        "latency_ms": round(gen.latency_ms, 2),
    }


# --- LLM tool-use loop (Claude) ---------------------------------------------------------------------

SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": "Hybrid search over the GitLab product documentation. Returns numbered "
    "excerpts with titles and URLs. Use it for any how-to, configuration or "
    "troubleshooting question.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


async def run_tool_use_loop(rt: Runtime, question: str, max_turns: int = 5) -> State:
    """Let the model pick MCP tools and doc searches, then answer with citations."""
    tools = [SEARCH_DOCS_TOOL] + [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in await rt.tools.list()
    ]
    messages: list[dict] = [{"role": "user", "content": question}]
    contexts: list[Candidate] = []
    tool_results: list[dict] = []
    calls: list[dict] = []
    flags: list[str] = []
    text = ""
    for _ in range(max_turns):
        t0 = time.perf_counter()
        resp = rt.llm_client.messages.create(
            model=rt.settings.anthropic_model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        calls.append(
            {
                "model": rt.settings.anthropic_model,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            break
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "search_docs":
                query = block.input.get("query", question)
                qvec = rt.embedder.embed([query])[0]
                with rt.pool.connection() as conn:
                    ranked = rt.reranker.score(query, hybrid_search(conn, query, qvec))
                start = len(contexts)
                new = ranked[: rt.settings.rerank_k]
                contexts += new
                content = "\n\n".join(
                    f"[{start + i}] {c.title} > {c.section}\nURL: {c.url}\n{c.text}"
                    for i, c in enumerate(new, 1)
                )
            else:
                result, flagged = _sanitize_result(await rt.tools.call(block.name, block.input))
                if flagged:
                    flags.append("tool_injection_removed")
                tool_results.append({"name": block.name, "args": block.input, "result": result})
                content = json.dumps(result, default=str)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        messages.append({"role": "user", "content": results})
    answered = bool(text) and NOT_FOUND not in text
    return {
        "intent": "llm_agent",
        "contexts": contexts,
        "tool_results": tool_results,
        "answer": text if answered else DECLINE_MSG,
        "answered": answered,
        "confidence": max((c.score for c in contexts), default=0.0),
        "llm_calls": calls,
        "flags": flags,
        "need_docs": bool(contexts),
    }


# --- Public entry point -------------------------------------------------------------------------------


def summarize(state: State, settings: Settings) -> dict:
    """API/eval-friendly response with citations, tools, timings and cost."""
    contexts = state.get("contexts", [])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", state.get("answer", ""))}
    citations = [
        {"n": i, "title": c.title, "section": c.section, "url": c.url}
        for i, c in enumerate(contexts, 1)
        if i in cited
    ]
    calls = state.get("llm_calls", [])
    inp = sum(c["input_tokens"] for c in calls)
    out = sum(c["output_tokens"] for c in calls)
    offline = all(c["model"] == "offline-extractive" for c in calls)
    ref = settings.cost_reference_model
    projected = cost_usd(ref, inp, out)
    if not calls and state.get("blocked"):
        inp = out = 0
    return {
        "answer": state.get("answer", ""),
        "answered": bool(state.get("answered")),
        "blocked": bool(state.get("blocked")),
        "block_reason": state.get("block_reason", ""),
        "intent": state.get("intent", ""),
        "confidence": round(float(state.get("confidence", 0.0)), 4),
        "coverage": round(float(state.get("coverage", 0.0)), 4),
        "unknown_terms": state.get("unknown_terms", []),
        "citations": citations,
        "tool_calls": [
            {"name": t["name"], "args": t["args"]} for t in state.get("tool_results", [])
        ],
        "guardrail_flags": state.get("flags", []),
        "timings_ms": state.get("timings", {}),
        "usage": {
            "model": calls[0]["model"] if calls else None,
            "input_tokens": inp,
            "output_tokens": out,
            "cost_usd": 0.0
            if offline
            else sum(cost_usd(c["model"], c["input_tokens"], c["output_tokens"]) for c in calls),
            "projected_cost_usd": projected if offline else None,
            "projected_model": ref if offline else None,
            "prices_per_mtok": PRICES_PER_MTOK.get(ref),
        },
        "_contexts": contexts,
        "_tool_results": state.get("tool_results", []),
        "_candidates": state.get("candidates", []),
        "_search_query": state.get("search_query"),
    }


__all__ = [
    "Runtime",
    "build_graph",
    "summarize",
    "plan_rules",
    "crm_filters",
    "DECLINE_MSG",
    "estimate_tokens",
    "build_user_message",
]
