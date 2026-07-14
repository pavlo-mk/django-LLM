from django.contrib import admin

from .models import Document, Message, Thread


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ("role", "content", "created_at")
    can_delete = False


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("__str__", "thread_id", "created_at", "updated_at")
    readonly_fields = ("thread_id", "created_at", "updated_at")
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("thread", "role", "created_at")
    list_filter = ("role",)
    search_fields = ("content",)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("source", "chunk_count", "char_count", "created_at")
    search_fields = ("source",)
