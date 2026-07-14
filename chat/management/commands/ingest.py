"""Ingest a file or directory of documents into the RAG knowledge base.

Usage:
    python manage.py ingest sample_docs/
    python manage.py ingest path/to/file.md
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from chat.ingestion import SUPPORTED_SUFFIXES, ingest, read_path


class Command(BaseCommand):
    help = "Ingest a file or directory of documents into the RAG knowledge base."

    def add_arguments(self, parser):
        parser.add_argument("path", help="File or directory to ingest")

    def handle(self, *args, **options):
        root = Path(options["path"])
        if not root.exists():
            raise CommandError(f"path does not exist: {root}")

        if root.is_file():
            files = [root]
        else:
            files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not files:
            raise CommandError(
                f"no ingestible files ({', '.join(sorted(SUPPORTED_SUFFIXES))}) under {root}"
            )

        total = 0
        for path in files:
            text = read_path(path)
            if not text.strip():
                self.stdout.write(self.style.WARNING(f"skip (empty): {path.name}"))
                continue
            doc = ingest(text, source=path.name)
            total += doc.chunk_count
            self.stdout.write(self.style.SUCCESS(f"ingested {path.name}: {doc.chunk_count} chunks"))

        self.stdout.write(self.style.SUCCESS(f"done — {len(files)} file(s), {total} chunks"))
