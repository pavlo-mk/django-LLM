import uuid

from django.db import models


class Thread(models.Model):
    """A conversation. ``thread_id`` is the key LangGraph's checkpointer uses,
    so the Django row and the checkpointed graph state stay in sync."""

    thread_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False, db_index=True)
    title = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title or f"Thread {self.thread_id}"


class Message(models.Model):
    """A single turn in a thread, mirrored into Django for admin + querying.

    The authoritative agent state lives in the LangGraph checkpointer; these
    rows are a human-readable copy for the admin and the UI."""

    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    thread = models.ForeignKey(Thread, related_name="messages", on_delete=models.CASCADE)
    role = models.CharField(max_length=16, choices=Role.choices)
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        preview = self.content[:40]
        return f"[{self.role}] {preview}"


class Document(models.Model):
    """Metadata about a source ingested into the RAG knowledge base.

    The actual embedded chunks live in the pgvector tables managed by
    langchain-postgres; this row is a human-readable record for the admin/UI."""

    source = models.CharField(max_length=255)
    char_count = models.PositiveIntegerField(default=0)
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.source} ({self.chunk_count} chunks)"
