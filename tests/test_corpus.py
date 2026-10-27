from enterprise_assistant.corpus import (
    MAX_WORDS,
    _windows,
    chunk_page,
    clean_markdown,
    reflow,
)

PAGE = """---
title: CI/CD variables
---

{{< details >}}

- Tier: Free, Premium, Ultimate

{{< /details >}}

{{< history >}}
- Introduced in GitLab 16.0.
{{< /history >}}

Intro paragraph that is wrapped
across two lines with a [link](other.md).

## Mask a CI/CD variable

You can mask a variable so its value is
hidden in job logs.

> [!note]
> Masking is not a guaranteed way
> to prevent leaks.

```yaml
job:
  script: echo "$SECRET"
```
"""


def test_clean_markdown_removes_shortcodes_keeps_tier_and_link_text():
    text = clean_markdown(PAGE)
    assert "{{<" not in text
    assert "Tier: Free, Premium, Ultimate" in text
    assert "Introduced in GitLab" not in text  # version history is noise
    assert "with a link." in text and "other.md" not in text


def test_reflow_joins_wrapped_lines_and_blockquotes_but_not_code():
    out = reflow("a wrapped\nline\n\n- item\n- item 2\n\n> one\n> two\n\n```\nx\ny\n```")
    assert "a wrapped line" in out
    assert "- item\n- item 2" in out
    assert "> one two" in out
    assert "```\nx\ny\n```" in out


def test_chunk_page_sections_urls_and_titles():
    chunks = chunk_page("doc/ci/variables/_index.md", PAGE)
    sections = {c.section for c in chunks}
    assert "Mask a CI/CD variable" in sections
    mask = next(c for c in chunks if c.section == "Mask a CI/CD variable")
    assert mask.title == "CI/CD variables"
    assert mask.url == "https://docs.gitlab.com/ci/variables/#mask-a-cicd-variable"
    assert "hidden in job logs" in mask.text


def test_windows_never_split_code_blocks():
    code = "```\n" + "\n".join(f"line {i}" for i in range(60)) + "\n```"
    body = "\n\n".join(["word " * 120, code, "word " * 120])
    for window in _windows(body):
        assert window.count("```") % 2 == 0
    assert all(len(w.split()) <= MAX_WORDS * 2 + 130 for w in _windows(body))
