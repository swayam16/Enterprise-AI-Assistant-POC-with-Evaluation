# Support assistant POC: results

**Recommendation: go with conditions.** The assistant fully met 9 of the 11 success criteria.
Ticket and CRM lookups were exact, no attack got through and no customer data leaked. Two gaps
remain: it declines out-of-scope questions 80% of the time against an 85% target, and the
documentation target with Claude writing the answers is still untested. Next step: score 50 of your real questions with Claude switched on,
then decide on the pilot.

## Results against the agreed criteria

Benchmark of 220 questions, scored automatically. Generator: offline (no LLM).

| Criterion | Target | Result | Status |
|---|---|---|---|
| Accuracy, all question types | ≥ 80% | 83.6% | Met |
| Accuracy, documentation questions | ≥ 70% offline | 72.7% | Met |
| Accuracy, ticket and CRM lookups | ≥ 95% | 100% (65 of 65) | Met |
| Answers backed by the cited source | ≥ 95% of sentences | 100% | Met |
| Right documentation section in top 5 | ≥ 85% | 91.5% | Met |
| Declines questions the sources don't cover | ≥ 85% | 80% (12 of 15) | Missed by 5 pts |
| Prompt injection and contact-detail requests blocked | 100% | 100% (8 of 8, plus 2 of 2 hidden in tickets) | Met |
| Answers leaking an email or phone number | 0 | 0 | Met |
| Response time, 95th percentile | ≤ 1.5 s | 0.12 s (before LLM time) | Met |
| Cost per question | ≤ $0.01 | $0.0032 projected on Claude Sonnet 5.5 | Met |
| Every metric automated and enforced in CI | Yes | Yes | Met |
| Accuracy with Claude generating answers | ≥ 85% docs | Not measured (no API key in POC) | Open |

On 35 held-out questions written after the system was frozen, accuracy was 74.3%, documentation
answers 76.7% and top-5 retrieval 96.7%. That is in line with the benchmark, which suggests the results are
not an artifact of tuning. It declined 3 of the 5 out-of-scope questions in that set.

## What this means for the support team

- **Lookups are ready.** Ticket status, owner, priority, renewal dates, ARR and at-risk account
  lists were correct every time.
- **Documentation answers are grounded but not yet complete enough.** About three in four contain every
  required fact. Every answer links to the exact section, so an engineer can
  verify it in one click. An LLM should lift completeness; that is what the next test measures.
- **The safety controls work.** Contact details are masked at the source, injected instructions in
  ticket text are stripped before the model sees them, and unsupported sentences are removed.
- **Cost is not a constraint.** At 2,000 questions a day, projected LLM spend is about $6.40 a day
  on Sonnet 5.5, or $0.32 on Haiku 5.5, at list prices.

## Where it falls short

- It still answers some questions that sit next to the documentation but aren't covered by it
  (for example, Pages domain verification, SLO error budgets and Jira Service Management). Fix: an LLM answerability check and a
  stricter coverage threshold, retested on the acceptance set.
- The documentation here is GitLab's public docs, standing in for yours. Results on your
  internal documentation may differ, which is why the acceptance set uses your real questions.

## Next steps

1. Week 1: you provide 50 anonymized questions from the queue; a support lead labels the answers.
2. Week 1: run the harness on them with Claude Sonnet 5.5 enabled; report against the same criteria.
3. Week 2: go/no-go on a 6-week pilot with 10 Tier 1 and Tier 2 engineers, read-only access.
