import re
from types import SimpleNamespace

from enterprise_assistant.agent import crm_filters, plan_rules, tool_evidence
from enterprise_assistant.guardrails import support_score
from enterprise_assistant.llm import (
    AnthropicGenerator,
    OfflineGenerator,
    answer_type,
    cost_usd,
    extract_answer,
    render_tool_answer,
    set_idf,
    unknown_terms,
)
from enterprise_assistant.search import Candidate, to_tsquery_text

ACCOUNTS = re.compile(r"\b(Juniper Retail|Kestrel Foods)\b", re.I)


def cand(text, title="Page", section="Section", score=0.9):
    return Candidate(
        "id",
        "doc/x.md",
        "https://docs.gitlab.com/x/",
        title,
        section,
        text,
        0.5,
        1,
        0.1,
        1,
        0.03,
        score,
    )


def test_plan_routes_by_intent():
    assert plan_rules("What's the status of SUP-1042?", ACCOUNTS)[0] == "ticket"
    intent, calls, need_docs = plan_rules("How should we fix SUP-1042?", ACCOUNTS)
    assert intent == "ticket_docs" and need_docs and calls[0]["name"] == "get_ticket"
    assert plan_rules("What plan is Juniper Retail on?", ACCOUNTS)[0] == "account"
    assert plan_rules("List the open tickets for Kestrel Foods.", ACCOUNTS)[0] == "tickets"
    assert (
        plan_rules("Which Ultimate accounts renew before 2027-01-31?", ACCOUNTS)[0] == "crm_query"
    )
    assert plan_rules("How do I mask a variable?", ACCOUNTS) == ("docs", [], True)


def test_crm_filters_parse_natural_language():
    f = crm_filters("Which accounts with a health score of 40 or lower have an unsolved P1 ticket?")
    assert f == {"max_health_score": 40, "has_unsolved_priority": "P1"}
    f = crm_filters("Which Premium accounts in EMEA renew before 2027-04-30?")
    assert f == {"plan": "Premium", "region": "EMEA", "renewal_before": "2027-04-30"}
    f = crm_filters("Which accounts with at least $500,000 ARR renew by March 31, 2027?")
    assert f == {"min_arr_usd": 500000, "renewal_before": "2027-03-31"}


def test_tsquery_ors_content_words_and_drops_stopwords():
    assert to_tsquery_text("How do I mask a CI/CD variable?") == "mask | ci | cd | variable"


def test_extract_answer_picks_the_fact_and_brings_list_items():
    text = (
        "Revoked tokens are kept for a while.\n"
        "By default, GitLab deletes revoked group access tokens after 30 days.\n"
        "The package filename cannot:\n- Start with a tilde\n- Include spaces"
    )
    ans, cited = extract_answer("How long are revoked group access tokens kept?", [cand(text)])
    assert "30 days" in ans and cited == [1]
    ans, _ = extract_answer("Can a package filename contain spaces?", [cand(text)])
    assert "Include spaces" in ans


def test_answer_type_forces_a_role_when_asked_who():
    want = answer_type("Who can cancel auto-merge?")
    assert want("You must have the Developer role.") and not want("Select Cancel.")


def test_unknown_terms_flags_words_the_corpus_never_saw():
    set_idf({"pipelin": 2.0, "gitlab": 1.0, "connect": 3.0, "incident": 4.0})
    assert unknown_terms({"pagerduty", "connect", "incident"}) == ["pagerduty"]
    set_idf({})


def test_tool_answers_are_grounded_in_tool_output():
    result = {
        "name": "Juniper Retail",
        "account_id": "ACC-101",
        "plan": "Premium",
        "seats": 50,
        "arr_usd": 17400,
        "region": "EMEA",
        "csm": "Grace Whitfield",
        "renewal_date": "2027-04-26",
        "health_score": 49,
        "unsolved_tickets_by_priority": {"P1": 0, "P2": 1, "P3": 0, "P4": 0},
    }
    tools = [{"name": "get_account", "args": {}, "result": result}]
    for q in (
        "What plan is Juniper Retail on and when do they renew?",
        "Who is the CSM for Juniper Retail and what is their ARR?",
        "How many unsolved P1 tickets does Juniper Retail have?",
    ):
        answer = render_tool_answer(q, tools)
        assert support_score(answer, [tool_evidence(result)]) >= 0.8, answer
    assert "0 unsolved P1 tickets" in render_tool_answer(
        "How many unsolved P1 tickets does Juniper Retail have?", tools
    )


def test_offline_generator_reports_projected_tokens():
    gen = OfflineGenerator().generate(
        "How do I mask a variable?", [cand("To mask a variable, select Mask variable.")], [], True
    )
    assert gen.found and gen.input_tokens > 50 and gen.model == "offline-extractive"


def test_cost_uses_published_prices():
    # Sonnet 5.5: $2 in / $10 out per million tokens.
    assert abs(cost_usd("claude-sonnet-5-5", 1_000_000, 100_000) - 3.0) < 1e-9


def test_anthropic_generator_uses_messages_api_and_usage():
    calls = {}

    class FakeMessages:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Select Mask variable [1].")],
                usage=SimpleNamespace(input_tokens=900, output_tokens=12),
            )

    gen = AnthropicGenerator("claude-sonnet-5-5", client=SimpleNamespace(messages=FakeMessages()))
    out = gen.generate("How do I mask a variable?", [cand("Select Mask variable.")], [], True)
    assert out.found and out.input_tokens == 900 and out.output_tokens == 12
    assert calls["model"] == "claude-sonnet-5-5" and "[1]" in calls["messages"][0]["content"]
