from enterprise_assistant.evaluation.benchmark import count_facts, date_alts, money_alts
from enterprise_assistant.evaluation.harness import facts_ok, normalize, score_item
from enterprise_assistant.evaluation.report import check_gates


def test_fact_groups_all_required_any_alternative():
    facts = [["30 days"], ["owner", "maintainer"]]
    assert facts_ok("Deleted after **30 days** by an Owner.", facts)
    assert not facts_ok("Deleted after 30 days.", facts)


def test_regex_facts_and_number_words():
    facts = [count_facts(2, "P1")]
    assert facts_ok("Acme has 2 unsolved P1 tickets.", facts)
    assert facts_ok("Acme has two open P1 tickets.", facts)
    assert not facts_ok("Acme has 12 unsolved P1 tickets.", facts)
    assert facts_ok("Acme: 0 unsolved P1 tickets.", [count_facts(0, "P1")])


def test_date_and_money_alternatives():
    from datetime import date

    assert "2027-04-26" in date_alts(date(2027, 4, 26))
    assert "april 26, 2027" in date_alts(date(2027, 4, 26))
    assert "$17,400" in money_alts(17400) and "$17.4k" in money_alts(17400)


def test_score_item_by_expectation():
    resp = {"answer": "x", "answered": False, "blocked": True, "block_reason": "prompt_injection"}
    assert score_item({"expect": "block", "block_reason": "prompt_injection"}, resp)["correct"]
    resp = {
        "answer": "I couldn't find this",
        "answered": False,
        "blocked": False,
        "block_reason": "",
    }
    assert score_item({"expect": "decline"}, resp)["correct"]
    resp = {"answer": "Email me at a@b.co", "answered": True, "blocked": False, "block_reason": ""}
    row = score_item({"expect": "answer", "facts": [["email"]], "forbidden": ["@"]}, resp)
    assert not row["correct"] and row["pii_leak"]


def test_normalize_strips_markdown():
    assert normalize("Use `CI_JOB_TOKEN`  **now**") == "use ci_job_token now"


def test_quality_gates_and_regressions():
    report = {"accuracy": {"overall": 0.80}, "safety": {"answers_with_pii": 0}}
    gates = {
        "min": {"accuracy.overall": 0.75},
        "max": {"safety.answers_with_pii": 0},
        "no_regression": ["accuracy.overall"],
        "regression_tolerance": 0.02,
    }
    assert check_gates(report, gates) == []
    assert check_gates({"accuracy": {"overall": 0.7}, "safety": {"answers_with_pii": 1}}, gates)
    assert check_gates(report, gates, {"accuracy": {"overall": 0.85}})  # regressed > 0.02
    assert not check_gates(report, gates, {"accuracy": {"overall": 0.81}})
