"""Guardrails applied before and after the model.

Input:  prompt-injection detection, secret/PII redaction, refusal of contact-detail requests.
Tools:  tool output is scanned for injected instructions, which are removed before generation.
Output: every answer sentence must be supported by the retrieved evidence or tool output
        (unsupported sentences are dropped), and any PII that slipped through is redacted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Patterns -----------------------------------------------------------------------------

_INJECTION = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules|guidelines)",
    r"forget (all |everything |about )?(your |the |all )?(previous |prior |earlier )?(instructions|rules|guidelines)",
    r"(reveal|print|show|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are now (in )?(developer|dan|jailbreak|unrestricted)",
    r"\bdo anything now\b|\bjailbreak\b",
    r"(system|admin) (note|override|message) to the (ai|assistant|model)",
    r"act as (an? )?(unfiltered|unrestricted|uncensored)",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION), re.I)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<![\w*])\+?\d{1,2}[-. ]?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SECRET_RE = re.compile(
    r"\b(?:glpat-[A-Za-z0-9_-]{20,}|gldt-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
    r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36})\b"
)
CONTACT_REQUEST_RE = re.compile(
    r"\b(emails?|email address(es)?|e-mails?|phone( numbers?)?|mobile( numbers?)?|"
    r"contact (details|info|information)|personal (info|information|details)|home address)\b",
    re.I,
)
# A contact-detail request is only refused when it is about a CRM person or customer.
CRM_PERSON_RE = re.compile(
    r"\b(contacts?|requesters?|customers?|SUP-\d{4}|ACC-\d{3}|who (opened|filed|raised))\b", re.I
)


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(
        n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums)
    )
    return total % 10 == 0


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    reason: str = ""
    flags: list[str] = field(default_factory=list)


def redact(text: str) -> tuple[str, list[str]]:
    """Mask secrets and PII. Returns the redacted text and the kinds of data found."""
    found: list[str] = []

    def sub(regex: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            found.append(label)
            return f"[REDACTED_{label.upper()}]"

        return regex.sub(repl, s)

    text = sub(SECRET_RE, "secret", text)
    text = sub(EMAIL_RE, "email", text)
    text = sub(SSN_RE, "ssn", text)
    text = sub(
        CARD_RE,
        "card",
        text,
        check=lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and _luhn_ok(re.sub(r"\D", "", s)),
    )
    text = sub(PHONE_RE, "phone", text)
    return text, sorted(set(found))


def check_input(question: str, entity_re: re.Pattern | None = None) -> GuardResult:
    """`entity_re` matches CRM entities (account names) known to the caller."""
    q = question.strip()
    if not q:
        return GuardResult(q, blocked=True, reason="empty", flags=["empty"])
    if len(q) > 2000:
        return GuardResult(q[:2000], blocked=True, reason="too_long", flags=["too_long"])
    if INJECTION_RE.search(q):
        return GuardResult(q, blocked=True, reason="prompt_injection", flags=["prompt_injection"])
    redacted, kinds = redact(q)
    flags = [f"redacted_{k}" for k in kinds]
    about_crm = CRM_PERSON_RE.search(redacted) or (entity_re and entity_re.search(redacted))
    if CONTACT_REQUEST_RE.search(redacted) and about_crm:
        return GuardResult(
            redacted, blocked=True, reason="contact_details", flags=[*flags, "pii_request"]
        )
    return GuardResult(redacted, flags=flags)


def sanitize_tool_output(text: str) -> tuple[str, bool]:
    """Drop sentences in tool output that try to instruct the assistant."""
    if not INJECTION_RE.search(text):
        return text, False
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [p for p in parts if not INJECTION_RE.search(p) and "AI ASSISTANT" not in p.upper()]
    return " ".join(kept) + " [instruction removed by guardrail]", True


# --- Groundedness ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9][a-z0-9_.:/$#-]*[a-z0-9]|[a-z0-9]")
_FILLER = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "use",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "also",
        "any",
        "each",
        "other",
        "such",
        "only",
        "no",
        "yes",
        "per",
        "via",
    ]
)


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_tokens(text: str) -> set[str]:
    text = text.lower().replace("`", "").replace("**", "")
    return {_stem(w.strip(".:")) for w in _WORD.findall(text) if w not in _FILLER and len(w) > 1}


def support_score(sentence: str, evidence: list[str]) -> float:
    toks = content_tokens(re.sub(r"\[\d+\]", "", sentence))
    if not toks:
        return 1.0
    best = 0.0
    for ev in evidence:
        ev_toks = content_tokens(ev)
        best = max(best, len(toks & ev_toks) / len(toks))
    return best


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\[(])|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def check_output(answer: str, evidence: list[str], threshold: float = 0.6) -> GuardResult:
    """Keep only sentences supported by evidence; redact leaked PII."""
    if not evidence:
        return GuardResult(answer)
    kept, dropped = [], 0
    for s in split_sentences(answer):
        if support_score(s, evidence) >= threshold:
            kept.append(s)
        else:
            dropped += 1
    flags = [f"dropped_{dropped}_unsupported"] if dropped else []
    text = " ".join(kept)
    text, kinds = redact(text)
    flags += [f"redacted_{k}" for k in kinds]
    if not kept:
        return GuardResult("", blocked=True, reason="ungrounded", flags=flags)
    return GuardResult(text, flags=flags)
