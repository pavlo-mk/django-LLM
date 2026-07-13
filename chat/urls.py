from django.urls import path

from . import views

app_name = "chat"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/threads/", views.create_thread, name="create_thread"),
    path("api/threads/<uuid:thread_id>/messages/", views.thread_messages, name="messages"),
    path("api/chat/", views.chat, name="chat"),
    path("api/threads/<uuid:thread_id>/stream/", views.stream, name="stream"),
]
