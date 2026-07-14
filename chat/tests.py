"""Tests for the tools and the Django views.

Agent/LLM calls are mocked so these run fast and offline (no Ollama needed).
An end-to-end smoke test that actually calls the model lives behind a flag.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.test import AsyncClient, TestCase
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessageChunk

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


# ---------------------------------------------------------------------------
# Unit tests that mock the Ollama / pgvector / DB boundaries, so the agent
# plumbing is exercised without live services.
# ---------------------------------------------------------------------------


class GraphUnitTests(TestCase):
    def test_token_helper(self):
        from agent.graph import _token

        self.assertEqual(_token(AIMessageChunk(content="hi")), "hi")
        self.assertEqual(_token("not a chunk"), "")
        self.assertEqual(_token(AIMessageChunk(content=[{"x": 1}])), "")

    @patch("agent.graph.ChatOllama")
    def test_build_llm(self, mock_chat):
        from agent import graph

        graph.build_llm()
        mock_chat.assert_called_once()

    @patch("agent.graph.create_react_agent", return_value="AGENT")
    @patch("agent.graph.get_checkpointer")
    @patch("agent.graph.build_llm")
    def test_get_agent(self, mock_llm, mock_ckpt, mock_cra):
        from agent import graph

        graph.get_agent.cache_clear()
        self.assertEqual(graph.get_agent(), "AGENT")
        mock_cra.assert_called_once()
        graph.get_agent.cache_clear()

    @patch("agent.graph.get_agent")
    def test_run(self, mock_get_agent):
        from agent import graph

        agent = MagicMock()
        agent.invoke.return_value = {"messages": [MagicMock(content="answer")]}
        mock_get_agent.return_value = agent
        self.assertEqual(graph.run("tid", "hi"), "answer")

    @patch("agent.graph.get_agent")
    def test_stream_tokens(self, mock_get_agent):
        from agent import graph

        agent = MagicMock()
        agent.stream.return_value = [
            (AIMessageChunk(content="a"), {}),
            ("not-a-chunk", {}),
            (AIMessageChunk(content="b"), {}),
        ]
        mock_get_agent.return_value = agent
        self.assertEqual(list(graph.stream_tokens("tid", "hi")), ["a", "b"])


async def test_get_async_agent(monkeypatch):
    from agent import graph

    graph._async_agent = None
    monkeypatch.setattr(graph, "build_llm", MagicMock(return_value="LLM"))
    monkeypatch.setattr(graph, "get_async_checkpointer", AsyncMock(return_value="CK"))
    monkeypatch.setattr(graph, "create_react_agent", MagicMock(return_value="AGENT"))
    assert await graph.get_async_agent() == "AGENT"
    graph._async_agent = None


async def test_astream_tokens(monkeypatch):
    from agent import graph

    class _FakeAgent:
        async def astream(self, _input, _config, stream_mode=None):
            for chunk in [AIMessageChunk(content="a"), "x", AIMessageChunk(content="b")]:
                yield chunk, {}

    monkeypatch.setattr(graph, "get_async_agent", AsyncMock(return_value=_FakeAgent()))
    out = [t async for t in graph.astream_tokens("tid", "hi")]
    assert out == ["a", "b"]


class RagStoreUnitTests(TestCase):
    @patch("agent.rag.OllamaEmbeddings")
    def test_get_embeddings(self, mock_emb):
        from agent import rag

        rag.get_embeddings.cache_clear()
        rag.get_embeddings()
        mock_emb.assert_called_once()
        rag.get_embeddings.cache_clear()

    @patch("agent.rag.PGVector")
    @patch("agent.rag.get_embeddings")
    def test_get_vectorstore(self, mock_emb, mock_pg):
        from agent import rag

        rag.get_vectorstore.cache_clear()
        rag.get_vectorstore()
        mock_pg.assert_called_once()
        rag.get_vectorstore.cache_clear()

    @patch("agent.rag.get_vectorstore")
    def test_ingest_text(self, mock_gv):
        from agent import rag

        store = MagicMock()
        mock_gv.return_value = store
        n = rag.ingest_text("hello world foo bar", source="s.txt")
        self.assertGreaterEqual(n, 1)
        store.add_documents.assert_called_once()

    @patch("agent.rag.get_vectorstore")
    def test_search(self, mock_gv):
        from agent import rag

        store = MagicMock()
        store.similarity_search.return_value = ["doc"]
        mock_gv.return_value = store
        self.assertEqual(rag.search("q"), ["doc"])


async def test_asearch(monkeypatch):
    from agent import rag

    monkeypatch.setattr(rag, "search", lambda q, k: ["d"])
    assert await rag.asearch("q") == ["d"]


async def test_arag_answer(monkeypatch):
    from agent import rag_graph

    async def fake_asearch(q, k=None):
        return [LCDocument(page_content="5 years", metadata={"source": "m"})]

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="5 years."))
    monkeypatch.setattr(rag_graph, "asearch", fake_asearch)
    monkeypatch.setattr(rag_graph, "build_llm", lambda: llm)
    rag_graph.get_rag_graph.cache_clear()

    res = await rag_graph.arag_answer("how long?")
    assert res["answer"] == "5 years."
    assert res["sources"] == ["m"]
    rag_graph.get_rag_graph.cache_clear()


async def test_arag_stream(monkeypatch):
    from agent import rag_graph

    async def fake_asearch(q, k=None):
        return [LCDocument(page_content="x", metadata={"source": "m"})]

    async def fake_astream(_prompt):
        for tok in ["5 ", "years."]:
            yield AIMessageChunk(content=tok)

    llm = MagicMock()
    llm.astream = fake_astream
    monkeypatch.setattr(rag_graph, "asearch", fake_asearch)
    monkeypatch.setattr(rag_graph, "build_llm", lambda: llm)

    kinds, toks = [], []
    async for kind, data in rag_graph.arag_stream("q"):
        kinds.append(kind)
        if kind == "token":
            toks.append(data)
    assert "sources" in kinds
    assert "".join(toks) == "5 years."


class CheckpointerUnitTests(TestCase):
    @patch("agent.checkpointer.PostgresSaver")
    @patch("agent.checkpointer.ConnectionPool")
    def test_get_checkpointer(self, mock_pool, mock_saver):
        from agent import checkpointer

        checkpointer._checkpointer = None
        checkpointer._pool = None
        saver = MagicMock()
        mock_saver.return_value = saver
        cp = checkpointer.get_checkpointer()
        saver.setup.assert_called_once()
        self.assertIs(cp, saver)
        checkpointer._checkpointer = None
        checkpointer._pool = None


async def test_get_async_checkpointer(monkeypatch):
    from agent import checkpointer

    checkpointer._async_checkpointer = None
    checkpointer._async_pool = None
    pool = MagicMock()
    pool.open = AsyncMock()
    saver = MagicMock()
    saver.setup = AsyncMock()
    monkeypatch.setattr(checkpointer, "AsyncConnectionPool", MagicMock(return_value=pool))
    monkeypatch.setattr(checkpointer, "AsyncPostgresSaver", MagicMock(return_value=saver))

    cp = await checkpointer.get_async_checkpointer()
    saver.setup.assert_awaited_once()
    assert cp is saver
    checkpointer._async_checkpointer = None
    checkpointer._async_pool = None


class IngestionReadTests(TestCase):
    def test_read_upload_text(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        from chat.ingestion import read_upload

        f = SimpleUploadedFile("note.txt", b"hello text", content_type="text/plain")
        self.assertEqual(read_upload(f), "hello text")

    def test_pdf_reading(self):
        import io
        import tempfile

        from django.core.files.uploadedfile import SimpleUploadedFile
        from pypdf import PdfWriter

        from chat.ingestion import read_path, read_upload

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        data = buf.getvalue()

        # upload branch → _read_pdf
        read_upload(SimpleUploadedFile("d.pdf", data, content_type="application/pdf"))
        # path branch → _read_pdf
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "d.pdf"
            p.write_bytes(data)
            read_path(p)


class IngestCommandEdgeTests(TestCase):
    def test_missing_path(self):
        from django.core.management import call_command
        from django.core.management.base import CommandError

        with self.assertRaises(CommandError):
            call_command("ingest", "/nope/does-not-exist")

    def test_no_ingestible_files(self):
        import tempfile

        from django.core.management import call_command
        from django.core.management.base import CommandError

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.bin").write_text("data", encoding="utf-8")
            with self.assertRaises(CommandError):
                call_command("ingest", d)

    @patch("chat.ingestion.ingest_text", return_value=0)
    def test_skips_empty_file(self, _mock):
        import tempfile
        from io import StringIO

        from django.core.management import call_command

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty.md").write_text("   ", encoding="utf-8")
            out = StringIO()
            call_command("ingest", d, stdout=out)
        self.assertIn("skip", out.getvalue())


class ModelStrTests(TestCase):
    def test_str_reprs(self):
        thread = Thread.objects.create()
        self.assertIn("Thread", str(thread))
        thread.title = "Greeting"
        self.assertEqual(str(thread), "Greeting")

        msg = Message.objects.create(thread=thread, role="user", content="hello there")
        self.assertIn("hello there", str(msg))

        doc = Document.objects.create(source="s.txt", chunk_count=3)
        self.assertIn("s.txt", str(doc))


class ViewSmokeTests(TestCase):
    def test_index(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_healthz(self):
        self.assertEqual(self.client.get("/healthz/").json()["status"], "ok")

    def test_thread_messages(self):
        thread = Thread.objects.create()
        Message.objects.create(thread=thread, role="user", content="hi")
        res = self.client.get(f"/api/threads/{thread.thread_id}/messages/")
        self.assertEqual(len(res.json()["messages"]), 1)
