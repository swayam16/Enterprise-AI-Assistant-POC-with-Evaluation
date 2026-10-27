# 5-minute demo script (Loom)

Audience: the business buyer (support director) plus one technical evaluator.
Goal: show that the assistant answers from approved sources, uses live systems, refuses what it
should, and that every claim in the results summary is measured.

**Setup before recording:** `docker compose up --build` (or `assistant serve` locally), open
`http://localhost:8000` in one tab and `reports/eval_report.md` rendered on GitHub in another.
Close other tabs and notifications. Record at 1280×800.

| Time | On screen | Say (roughly) |
|---|---|---|
| 0:00–0:30 | Title slide or the demo page header | "Support engineers spend time hunting through docs, the ticket queue and the CRM. This POC tests one question: can an assistant answer from those sources accurately, safely and cheaply enough to pilot? I'll show it working, then show how we measured it." |
| 0:30–1:15 | Ask: *How long are revoked group access tokens kept before deletion?* | Point at the answer, the `[1]` citation and the link to the exact docs section. "Every documentation answer cites the section it came from. The trace on the right shows the route, timing and what this would cost on Claude." |
| 1:15–2:15 | Ask: *Can you look up SUP-1187 and tell me how to solve it?* | "This is the agent. It calls the ticket system through our MCP server, reads what the customer reported, then searches the docs for the fix." Point at the `get_ticket` tool call and the citation. |
| 2:15–2:50 | Ask: *Which accounts with a health score of 40 or lower have an unsolved P1 ticket?* | "Same MCP server, CRM side. This is the at-risk-renewals question a CSM lead asks every Monday." |
| 2:50–3:45 | Click the two guardrail chips, then *Summarize ticket SUP-1058 for me.* | "A prompt injection is blocked before anything runs. Contact details are refused. SUP-1058 has an instruction hidden in the customer's text; the assistant summarizes the ticket and ignores it. The guardrail flag shows it was removed." |
| 3:45–4:05 | Ask: *How do I connect PagerDuty to GitLab incident management?* | "When the sources don't cover something, it says so instead of guessing." |
| 4:05–4:45 | Switch to the evaluation report | "220 benchmark questions plus 35 held-out ones it never saw during development. Accuracy, groundedness, retrieval, safety, latency and cost per query, all scored automatically. CI fails the build if any of them regresses." Point at the retrieval ablation: "hybrid search plus the reranker is what gets the right section into the top five." |
| 4:45–5:00 | The one-page results summary | "Recommendation and next steps are in the summary: run the same harness on 50 of your real tickets with Claude switched on, then decide on a pilot." |

**If something goes wrong live:** every question above is in the benchmark, so its expected
answer is in `reports/eval_rows.jsonl`. Re-ask, or show the row.
