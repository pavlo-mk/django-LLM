"""Tests for the tools and the Django views.

Agent/LLM calls are mocked so these run fast and offline (no Ollama needed).
An end-to-end smoke test that actually calls the model lives behind a flag.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from django.test import AsyncClient, TestCase
from langchain_core.documents import Document as LCDocument

from agent.tools import add, current_time, multiply, word_count

from .models import Document, Message, Thread


class ToolTests(TestCase):
    def test_add(self):
        self.assertEqual(add.invoke({"a": 2, "b": 3}), 5)

    def test_multiply(self):
        self.assertEqual(multiply.invoke({"a": 4, "b": 5}), 20)

    def test_word_count(self):
        self.assertEqual(word_count.invoke({"text": "one two three"}), 3)

    def test_current_time_is_iso(self):
        self.assertIn("T", current_time.invoke({}))


class ChatViewTests(TestCase):
    def test_create_thread(self):
        res = self.client.post("/api/threads/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("thread_id", res.json())
        self.assertEqual(Thread.objects.count(), 1)

    @patch("agent.graph.run", return_value="mocked reply")
    def test_chat_persists_user_and_assistant(self, mock_run):
        thread = Thread.objects.create()
        res = self.client.post(
            "/api/chat/",
            data={"thread_id": str(thread.thread_id), "message": "hi"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["reply"], "mocked reply")
        mock_run.assert_called_once()
        roles = list(thread.messages.values_list("role", flat=True))
        self.assertEqual(roles, ["user", "assistant"])
        # a fresh thread gets a title from its first message
        thread.refresh_from_db()
        self.assertEqual(thread.title, "hi")

    def test_chat_requires_message(self):
        thread = Thread.objects.create()
        res = self.client.post(
            "/api/chat/",
            data={"thread_id": str(thread.thread_id), "message": ""},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)


@pytest.mark.django_db(transaction=True)
async def test_stream_emits_sse_and_persists(monkeypatch):
    """The async SSE endpoint streams tokens and persists the full reply."""

    async def fake_astream(thread_id, message):
        for token in ["Hel", "lo"]:
            yield token

    monkeypatch.setattr("agent.graph.astream_tokens", fake_astream)

    thread = await Thread.objects.acreate()
    client = AsyncClient()
    res = await client.get(f"/api/threads/{thread.thread_id}/stream/?message=hi")

    body = b""
    async for chunk in res.streaming_content:
        body += chunk
    text = body.decode()

    assert res["Content-Type"] == "text/event-stream"
    assert 'data: "Hel"' in text
    assert 'data: "lo"' in text
    assert "event: done" in text

    # the concatenated stream is saved as one assistant message
    assistant = await Message.objects.aget(thread=thread, role=Message.Role.ASSISTANT)
    assert assistant.content == "Hello"


class RagUnitTests(TestCase):
    """RAG pieces with the vector store / embeddings / LLM mocked out."""

    def test_format_docs_tags_sources(self):
        from agent.rag import format_docs

        out = format_docs(
            [LCDocument(page_content="Warranty is 5 years.", metadata={"source": "m.txt"})]
        )
        self.assertIn("5 years", out)
        self.assertIn("source: m.txt", out)

    def test_format_docs_empty(self):
        from agent.rag import format_docs

        self.assertIn("No relevant", format_docs([]))

    @patch("agent.rag.search")
    def test_retriever_tool(self, mock_search):
        from agent.rag import search_knowledge_base

        mock_search.return_value = [
            LCDocument(page_content="Brew time is 7 seconds.", metadata={"source": "z.md"})
        ]
        out = search_knowledge_base.invoke({"query": "brew time"})
        self.assertIn("7 seconds", out)
        self.assertIn("source: z.md", out)

    def test_build_rag_prompt(self):
        from agent.rag_graph import build_rag_prompt

        prompt = build_rag_prompt(
            "How long?", [LCDocument(page_content="5 years", metadata={"source": "m"})]
        )
        self.assertIn("5 years", prompt)
        self.assertIn("How long?", prompt)


class IngestionTests(TestCase):
    def test_read_path_text(self):
        import tempfile

        from chat.ingestion import read_path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "note.md"
            p.write_text("# Hello\nbody text", encoding="utf-8")
            self.assertEqual(read_path(p), "# Hello\nbody text")

    @patch("chat.ingestion.ingest_text", return_value=3)
    def test_ingest_view_creates_document(self, mock_ingest):
        res = self.client.post("/api/ingest/", data={"text": "hello world", "source": "note"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"source": "note", "chunks": 3})
        mock_ingest.assert_called_once()
        self.assertEqual(Document.objects.get().chunk_count, 3)

    def test_ingest_view_requires_content(self):
        res = self.client.post("/api/ingest/", data={"text": ""})
        self.assertEqual(res.status_code, 400)

    @patch("chat.ingestion.ingest_text", return_value=2)
    def test_ingest_command(self, mock_ingest, tmp_path=None):
        import tempfile
        from io import StringIO

        from django.core.management import call_command

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "doc.md").write_text("# Title\nsome content", encoding="utf-8")
            out = StringIO()
            call_command("ingest", d, stdout=out)
        self.assertIn("doc.md", out.getvalue())
        self.assertEqual(Document.objects.get().source, "doc.md")


@pytest.mark.django_db(transaction=True)
async def test_rag_stream_emits_sources_and_answer(monkeypatch):
    """The RAG SSE endpoint emits a sources event then streams the answer."""

    async def fake_rag(question):
        yield "sources", ["m.txt"]
        yield "token", "5 years."

    monkeypatch.setattr("agent.rag_graph.arag_stream", fake_rag)

    thread = await Thread.objects.acreate()
    client = AsyncClient()
    res = await client.get(f"/api/threads/{thread.thread_id}/rag/?message=warranty")

    body = b""
    async for chunk in res.streaming_content:
        body += chunk
    text = body.decode()

    assert "event: sources" in text
    assert '["m.txt"]' in text
    assert 'data: "5 years."' in text
    assert "event: done" in text
    assistant = await Message.objects.aget(thread=thread, role=Message.Role.ASSISTANT)
    assert assistant.content == "5 years."


class AgentSmokeTest(TestCase):
    """Real end-to-end runs against Ollama + pgvector. Skipped unless
    RUN_AGENT_TESTS=1 (needs `ollama serve`, the models pulled, and Postgres)."""

    def test_agent_uses_tool(self):
        if os.environ.get("RUN_AGENT_TESTS") != "1":
            self.skipTest("set RUN_AGENT_TESTS=1 to run the live Ollama test")
        from agent import graph

        thread = Thread.objects.create()
        reply = graph.run(str(thread.thread_id), "What is 6 times 7? Use a tool.")
        self.assertIn("42", reply)

    def test_rag_end_to_end(self):
        if os.environ.get("RUN_AGENT_TESTS") != "1":
            self.skipTest("set RUN_AGENT_TESTS=1 to run the live RAG test")
        import uuid

        from agent.graph import run
        from agent.rag import ingest_text

        ingest_text("The Qwix gadget has a nine year warranty.", source="qwix.txt")
        reply = run(
            str(uuid.uuid4()),
            "What is the Qwix warranty? Search the knowledge base.",
        )
        self.assertTrue("nine" in reply.lower() or "9" in reply)
