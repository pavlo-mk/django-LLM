import json

from django.conf import settings
from django.http import (
    Http404,
    HttpResponseBadRequest,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from agent import graph, rag_graph

from . import ingestion
from .models import Message, Thread


@ensure_csrf_cookie
def index(request):
    """Render the chat page and set the CSRF cookie for the JS client."""
    threads = Thread.objects.all()[:20]
    return render(
        request,
        "chat/index.html",
        {"threads": threads, "model": settings.OLLAMA_MODEL},
    )


@require_GET
def healthz(request):
    """Liveness probe used by the container healthcheck."""
    return JsonResponse({"status": "ok"})


@require_POST
def create_thread(request):
    thread = Thread.objects.create()
    return JsonResponse({"thread_id": str(thread.thread_id)})


@require_POST
def ingest_document(request):
    """Ingest an uploaded file or pasted text into the RAG knowledge base."""
    upload = request.FILES.get("file")
    if upload is not None:
        source = upload.name or "upload"
        text = ingestion.read_upload(upload)
    else:
        text = (request.POST.get("text") or "").strip()
        source = request.POST.get("source") or "pasted-text"
    if not text.strip():
        return HttpResponseBadRequest("a non-empty file or text is required")

    doc = ingestion.ingest(text, source=source)
    return JsonResponse({"source": doc.source, "chunks": doc.chunk_count})


@require_GET
def thread_messages(request, thread_id):
    thread = get_object_or_404(Thread, thread_id=thread_id)
    data = [{"role": m.role, "content": m.content} for m in thread.messages.all()]
    return JsonResponse({"messages": data})


@require_POST
def chat(request):
    """Non-streaming turn: run the agent to completion and return the reply."""
    payload = json.loads(request.body or "{}")
    thread_id = payload.get("thread_id")
    text = (payload.get("message") or "").strip()
    if not thread_id or not text:
        return HttpResponseBadRequest("thread_id and message are required")

    thread = get_object_or_404(Thread, thread_id=thread_id)
    Message.objects.create(thread=thread, role=Message.Role.USER, content=text)

    reply = graph.run(str(thread.thread_id), text)

    Message.objects.create(thread=thread, role=Message.Role.ASSISTANT, content=reply)
    _touch_title(thread, text)
    return JsonResponse({"reply": reply})


async def stream(request, thread_id):
    """Streaming turn over Server-Sent Events (async, consumed by EventSource)."""
    if request.method != "GET":
        return HttpResponseBadRequest("GET only")
    text = (request.GET.get("message") or "").strip()
    if not text:
        return HttpResponseBadRequest("message is required")

    try:
        thread = await Thread.objects.aget(thread_id=thread_id)
    except Thread.DoesNotExist as exc:
        raise Http404("thread not found") from exc

    await Message.objects.acreate(thread=thread, role=Message.Role.USER, content=text)
    await _atouch_title(thread, text)

    async def event_stream():
        collected: list[str] = []
        try:
            async for token in graph.astream_tokens(str(thread.thread_id), text):
                collected.append(token)
                yield _sse(token)
        finally:
            # Persist whatever we produced, even if the client disconnects.
            await Message.objects.acreate(
                thread=thread,
                role=Message.Role.ASSISTANT,
                content="".join(collected),
            )
        yield _sse("", event="done")

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # disable proxy buffering
    return response


async def rag_stream(request, thread_id):
    """Streaming turn using the dedicated retrieve→generate RAG pipeline.

    Emits a ``sources`` event (the retrieved document names) before the answer
    tokens, so the UI can show what the answer was grounded in.
    """
    if request.method != "GET":
        return HttpResponseBadRequest("GET only")
    text = (request.GET.get("message") or "").strip()
    if not text:
        return HttpResponseBadRequest("message is required")

    try:
        thread = await Thread.objects.aget(thread_id=thread_id)
    except Thread.DoesNotExist as exc:
        raise Http404("thread not found") from exc

    await Message.objects.acreate(thread=thread, role=Message.Role.USER, content=text)
    await _atouch_title(thread, text)

    async def event_stream():
        collected: list[str] = []
        try:
            async for kind, data in rag_graph.arag_stream(text):
                if kind == "sources":
                    yield _sse(data, event="sources")
                else:
                    collected.append(str(data))
                    yield _sse(str(data))
        finally:
            await Message.objects.acreate(
                thread=thread,
                role=Message.Role.ASSISTANT,
                content="".join(collected),
            )
        yield _sse("", event="done")

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _sse(data: object, event: str | None = None) -> str:
    """Format one Server-Sent Event. Data is JSON-encoded to stay single-line."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


def _touch_title(thread: Thread, text: str) -> None:
    """Give a fresh thread a title from its first message and bump updated_at."""
    if not thread.title:
        thread.title = text[:80]
    thread.save()


async def _atouch_title(thread: Thread, text: str) -> None:
    if not thread.title:
        thread.title = text[:80]
    await thread.asave()
