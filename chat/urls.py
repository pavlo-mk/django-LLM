from django.urls import path

from . import views

app_name = "chat"

urlpatterns = [
    path("", views.index, name="index"),
    path("healthz/", views.healthz, name="healthz"),
    path("api/threads/", views.create_thread, name="create_thread"),
    path("api/ingest/", views.ingest_document, name="ingest"),
    path("api/threads/<uuid:thread_id>/messages/", views.thread_messages, name="messages"),
    path("api/chat/", views.chat, name="chat"),
    path("api/threads/<uuid:thread_id>/stream/", views.stream, name="stream"),
    path("api/threads/<uuid:thread_id>/rag/", views.rag_stream, name="rag_stream"),
]
