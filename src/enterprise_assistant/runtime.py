"""Wire the pieces together: DB pool, embedder, reranker, MCP tool client, generator, graph."""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client, StdioServerParameters
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .agent import Runtime, build_graph, summarize
from .config import Settings, get_settings
from .embeddings import load_embedder
from .llm import make_generator, set_idf
from .rerank import Reranker


class ToolBox:
    """Thin async wrapper around an MCP client session."""

    def __init__(self, client: Client):
        self.client = client
        self._tools: list[dict] | None = None

    async def list(self) -> list[dict]:
        if self._tools is None:
            res = await self.client.list_tools()
            self._tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in res.tools
            ]
        return self._tools

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        res = await self.client.call_tool(name, args)
        if getattr(res, "is_error", False):
            text = " ".join(getattr(c, "text", "") for c in res.content)
            return {"error": text or f"{name} failed"}
        return res.structured_content or {}


def mcp_params(settings: Settings) -> StdioServerParameters:
    env = {**os.environ, "DATABASE_URL": settings.database_url}
    return StdioServerParameters(
        command=sys.executable, args=["-m", "enterprise_assistant.mcp_server"], env=env
    )


class Assistant:
    """Async context manager owning every long-lived resource."""

    def __init__(
        self,
        settings: Settings | None = None,
        mcp_target: Any = None,
        generator=None,
        llm_client=None,
    ):
        self.settings = settings or get_settings()
        self._mcp_target = mcp_target
        self._generator = generator
        self._llm_client = llm_client
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Assistant:
        s = self.settings
        self.pool = ConnectionPool(
            s.database_url, min_size=1, max_size=8, configure=register_vector, open=True
        )
        self._stack.callback(self.pool.close)
        client = Client(self._mcp_target or mcp_params(s))
        await self._stack.enter_async_context(client)
        self.tools = ToolBox(client)
        accounts = await self.tools.call("query_accounts", {"limit": 50})
        idf_path = s.artifacts_dir / "idf.json"
        if idf_path.exists():
            set_idf(json.loads(idf_path.read_text()))
        llm_client = self._llm_client
        if llm_client is None and s.llm_provider == "anthropic":
            import anthropic

            llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.runtime = Runtime(
            settings=s,
            embedder=load_embedder(s.embedding_provider, s.artifacts_dir, s.embedding_dim),
            reranker=Reranker.load(s.artifacts_dir / "reranker.joblib"),
            pool=self.pool,
            tools=self.tools,
            generator=self._generator
            or make_generator(s.llm_provider, s.anthropic_model, s.openai_model),
            llm_client=llm_client,
            account_names=[a["name"] for a in accounts.get("accounts", [])],
        )
        self.graph = build_graph(self.runtime)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def ask(self, question: str) -> dict:
        state = await self.graph.ainvoke({"question": question, "timings": {}})
        return summarize(state, self.settings)
