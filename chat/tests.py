"""Tests for the tools and the Django views.

Agent/LLM calls are mocked so these run fast and offline (no Ollama needed).
An end-to-end smoke test that actually calls the model lives behind a flag.
"""

import os
from unittest.mock import patch

from django.test import TestCase

from agent.tools import add, current_time, multiply, word_count

from .models import Message, Thread


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

    @patch("agent.graph.stream_tokens", return_value=iter(["Hel", "lo"]))
    def test_stream_emits_sse_and_persists(self, mock_stream):
        thread = Thread.objects.create()
        res = self.client.get(f"/api/threads/{thread.thread_id}/stream/?message=hi")
        body = b"".join(res.streaming_content).decode()
        self.assertEqual(res["Content-Type"], "text/event-stream")
        self.assertIn('data: "Hel"', body)
        self.assertIn('data: "lo"', body)
        self.assertIn("event: done", body)
        # the concatenated stream is saved as one assistant message
        assistant = thread.messages.get(role=Message.Role.ASSISTANT)
        self.assertEqual(assistant.content, "Hello")


class AgentSmokeTest(TestCase):
    """Real end-to-end run against Ollama. Skipped unless RUN_AGENT_TESTS=1
    (it needs `ollama serve` up and the model pulled)."""

    def test_agent_uses_tool(self):
        if os.environ.get("RUN_AGENT_TESTS") != "1":
            self.skipTest("set RUN_AGENT_TESTS=1 to run the live Ollama test")
        from agent import graph

        thread = Thread.objects.create()
        reply = graph.run(str(thread.thread_id), "What is 6 times 7? Use a tool.")
        self.assertIn("42", reply)
