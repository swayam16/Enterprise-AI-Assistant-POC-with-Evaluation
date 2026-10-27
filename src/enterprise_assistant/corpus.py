"""Fetch the public corpus (GitLab product docs) and turn it into retrievable chunks.

Each Markdown page is cleaned of Hugo shortcodes and front matter, split on headings into
sections, and long sections are split into overlapping windows. Every chunk keeps its page
title, heading path and a public docs.gitlab.com URL so answers can cite it.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CORPUS_COMMIT, CORPUS_PATHS, CORPUS_REPO, DOCS_BASE_URL

MAX_WORDS = 220
OVERLAP_WORDS = 40
MIN_WORDS = 12


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str  # repo-relative path, e.g. doc/ci/variables/_index.md
    url: str
    title: str
    section: str  # heading path, "H2 > H3"
    text: str

    @property
    def section_key(self) -> str:
        return f"{self.doc_id}#{self.section}"


def fetch_docs(dest: Path, commit: str = CORPUS_COMMIT) -> Path:
    """Shallow, blob-less, sparse fetch of just the doc folders we index, pinned to one commit
    (a few MB instead of the whole GitLab repository)."""
    if (dest / ".git").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dest), *args], check=True)

    git("init", "--quiet")
    git("remote", "add", "origin", CORPUS_REPO)
    git("sparse-checkout", "set", *CORPUS_PATHS)
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "--quiet", "FETCH_HEAD")
    return dest


# --- Markdown cleaning -------------------------------------------------------------------

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_HISTORY = re.compile(r"\{\{<\s*history\s*>\}\}.*?\{\{<\s*/history\s*>\}\}", re.S)
_DETAILS = re.compile(r"\{\{<\s*details\s*>\}\}(.*?)\{\{<\s*/details\s*>\}\}", re.S)
_TAB = re.compile(r'\{\{<\s*tab\s+title="([^"]*)"\s*>\}\}')
_SHORTCODE = re.compile(r"\{\{<\s*/?\s*[a-z-]+[^>]*>\}\}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?(?:div|span|br|img|kbd|code|details|summary|a)[^>]*>", re.I)
_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*(?:\{#[^}]*\})?\s*$")


def _front_matter_title(raw: str) -> str:
    m = _FRONT_MATTER.match(raw)
    if not m:
        return ""
    for line in m.group(1).splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


_BLOCK_START = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s|\||#|>|```|:::|\[\^)")


def reflow(text: str) -> str:
    """Join hard-wrapped paragraph lines (the docs wrap at ~80 chars) so sentences stay whole.
    List items, tables, headings, quotes and code blocks keep their line structure."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        prev = out[-1] if out else ""
        # Blockquote/admonition continuation: "> text" following "> text" is one paragraph.
        if (
            not in_fence
            and stripped.startswith(">")
            and prev.strip().startswith(">")
            and not prev.strip().lstrip("> ").startswith("[!")
            and stripped.lstrip("> ").strip()
            and not stripped.lstrip("> ").startswith(("[!", "-", "*", "1."))
        ):
            out[-1] = prev.rstrip() + " " + stripped.lstrip("> ").strip()
            continue
        if (
            not in_fence
            and stripped
            and prev.strip()
            and not _BLOCK_START.match(line)
            and not prev.strip().startswith(("|", "#", "```"))
        ):
            out[-1] = prev.rstrip() + " " + stripped
        else:
            out.append(line)
    return "\n".join(out)


def clean_markdown(raw: str) -> str:
    text = _FRONT_MATTER.sub("", raw)
    text = _HTML_COMMENT.sub("", text)
    text = _HISTORY.sub("", text)
    # Keep the tier/offering line ("Tier: Premium, Ultimate"): it answers real buyer questions.
    text = _DETAILS.sub(lambda m: " ".join(m.group(1).replace("- ", "").split()) + "\n", text)
    text = _TAB.sub(lambda m: f"\n{m.group(1)}:\n", text)
    text = text.replace("{{< yes >}}", "Yes").replace("{{< no >}}", "No")
    text = _SHORTCODE.sub("", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return reflow(text).strip()


def _doc_url(doc_id: str) -> str:
    path = doc_id.removeprefix("doc/").removesuffix(".md")
    path = path.removesuffix("/_index").removesuffix("_index")
    return f"{DOCS_BASE_URL}/{path}/".replace("//", "/").replace("https:/", "https://")


def _anchor(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9 _-]", "", heading.lower().replace("`", ""))
    return re.sub(r"\s+", "-", slug.strip())


def split_sections(text: str, title: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body)] splitting on H2-H4. Code fences are never split."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m and len(m.group(1)) >= 2:
            level = len(m.group(1))
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, m.group(2).strip())]
            sections.append((" > ".join(h for _, h in stack), []))
        elif m:  # H1 duplicates the title
            continue
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _blocks(body: str) -> list[str]:
    """Split a section into blocks: paragraphs, list runs, tables and whole code fences."""
    blocks, cur, in_fence = [], [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            if not in_fence and cur:
                blocks.append("\n".join(cur))
                cur = []
            cur.append(line)
            in_fence = not in_fence
            if not in_fence:
                blocks.append("\n".join(cur))
                cur = []
            continue
        if not in_fence and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _nwords(text: str) -> int:
    return len(text.split())


def _windows(body: str) -> list[str]:
    """Pack whole blocks into windows of about MAX_WORDS, never cutting a code block or list
    item in half. Consecutive windows share their boundary block for context."""
    out: list[str] = []
    cur: list[str] = []
    for block in _blocks(body):
        if cur and sum(map(_nwords, cur)) + _nwords(block) > MAX_WORDS:
            out.append("\n\n".join(cur))
            tail = cur[-1]
            cur = [tail] if _nwords(tail) <= OVERLAP_WORDS else []
        if _nwords(block) > MAX_WORDS * 2 and not block.lstrip().startswith("```"):
            words = block.split(" ")  # a very long prose block: fall back to word windows
            for i in range(0, len(words), MAX_WORDS - OVERLAP_WORDS):
                piece = " ".join(words[i : i + MAX_WORDS])
                if cur:
                    out.append("\n\n".join(cur))
                    cur = []
                out.append(piece)
            continue
        cur.append(block)
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_page(doc_id: str, raw: str) -> list[Chunk]:
    title = _front_matter_title(raw) or Path(doc_id).stem.replace("_", " ")
    url = _doc_url(doc_id)
    chunks = []
    for heading, body in split_sections(clean_markdown(raw), title):
        # Section text keeps its own line structure (lists, tables, code) but windows by words.
        for i, win in enumerate(_windows(body)):
            text = win.strip()
            if len(text.split()) < MIN_WORDS:
                continue
            key = f"{doc_id}#{heading}#{i}"
            chunks.append(
                Chunk(
                    chunk_id=hashlib.sha1(key.encode()).hexdigest()[:16],
                    doc_id=doc_id,
                    url=url + (f"#{_anchor(heading.split(' > ')[-1])}" if heading else ""),
                    title=title,
                    section=heading,
                    text=text,
                )
            )
    return chunks


def build_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted((docs_dir / "doc").rglob("*.md")):
        doc_id = str(path.relative_to(docs_dir))
        chunks.extend(chunk_page(doc_id, path.read_text(encoding="utf-8")))
    seen, unique = set(), []
    for c in chunks:  # identical windows can repeat across tabs; keep the first
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)
    return unique


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
