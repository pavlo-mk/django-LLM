"""Load chat history into a pandas DataFrame and print a few summaries.

A playground for poking at the message history with pandas/numpy — nothing
here is used by the app itself.

Usage:
    python manage.py dfstats
    python manage.py dfstats --days 30 --top 5
    python manage.py dfstats --demo 500        # synthetic rows, no DB needed
    python manage.py dfstats --csv messages.csv
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from chat.models import Message

# ``thread_id`` below is the Message FK column (Thread's pk), not the LangGraph
# UUID on Thread.thread_id — renamed to "thread" in the frame to avoid the clash.
FIELDS = ["id", "thread_id", "thread__title", "role", "content", "created_at"]

BLOCKS = "▁▂▃▄▅▆▇█"

DEMO_WORDS = [
    "agent", "graph", "node", "state", "prompt", "token", "context", "vector",
    "chunk", "embed", "retrieve", "summarize", "explain", "error", "retry",
    "latency", "model", "thread", "message",
]  # fmt: skip


def load_frame(days: int | None) -> pd.DataFrame:
    """Pull messages out of the DB into a DataFrame."""
    qs = Message.objects.select_related("thread")
    if days is not None:
        qs = qs.filter(created_at__gte=timezone.now() - timedelta(days=days))
    df = pd.DataFrame.from_records(qs.values(*FIELDS))
    if df.empty:
        return df
    return df.rename(columns={"thread_id": "thread", "thread__title": "title"})


def demo_frame(rows: int, days: int, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic frame so the command is fun to run on an empty DB."""
    rng = np.random.default_rng(seed)
    threads = max(1, rows // 8)

    # Assistant turns run longer than user turns, hence the two length scales.
    role = np.where(np.arange(rows) % 2 == 0, "user", "assistant")
    scale = np.where(role == "user", 1.0, 2.2)
    lengths = np.clip(rng.lognormal(mean=2.2, sigma=0.6, size=rows) * scale, 1, None)

    offsets = np.sort(rng.uniform(0, days * 24 * 3600, size=rows))
    created = pd.Timestamp.utcnow().floor("s") - pd.to_timedelta(
        days * 24 * 3600 - offsets, unit="s"
    )

    thread = rng.integers(1, threads + 1, size=rows)
    return pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "thread": thread,
            "title": [f"demo thread {t}" for t in thread],
            "role": role,
            "content": [" ".join(rng.choice(DEMO_WORDS, size=int(n))) for n in lengths],
            "created_at": created,
        }
    )


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns every summary below leans on."""
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["chars"] = df["content"].str.len()
    df["words"] = df["content"].str.split().str.len()
    df["hour"] = df["created_at"].dt.hour
    return df


def sparkline(values: np.ndarray) -> str:
    """Scale a series into the block characters, tallest bucket = full block."""
    if values.size == 0:
        return ""
    top = float(values.max())
    if top == 0:
        return BLOCKS[0] * values.size
    idx = np.clip(np.rint(values / top * (len(BLOCKS) - 1)).astype(int), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[i] for i in idx)


class Command(BaseCommand):
    help = "Summarize chat history with pandas/numpy."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, help="Only messages from the last N days")
        parser.add_argument("--top", type=int, default=10, help="Busiest threads to show")
        parser.add_argument(
            "--demo",
            type=int,
            metavar="ROWS",
            help="Use N synthetic rows instead of the database",
        )
        parser.add_argument("--csv", help="Also write the frame to this path")

    def handle(self, *args, **options):
        if options["demo"]:
            df = demo_frame(options["demo"], days=options["days"] or 30)
            source = f"{options['demo']} synthetic rows"
        else:
            df = load_frame(options["days"])
            source = "database"

        if df.empty:
            raise CommandError("no messages to summarize — try --demo 500")

        df = enrich(df)
        self.stdout.write(self.style.SUCCESS(f"{len(df)} messages from {source}"))
        self.stdout.write(
            f"span: {df['created_at'].min():%Y-%m-%d} → {df['created_at'].max():%Y-%m-%d}"
            f"  ·  {df['thread'].nunique()} threads\n"
        )

        self._by_role(df)
        self._busiest_threads(df, options["top"])
        self._activity(df)

        if options["csv"]:
            path = Path(options["csv"])
            df.to_csv(path, index=False)
            self.stdout.write(self.style.SUCCESS(f"\nwrote {len(df)} rows to {path}"))

    def _by_role(self, df: pd.DataFrame) -> None:
        stats = df.groupby("role")["chars"].agg(
            messages="count",
            mean="mean",
            median="median",
            p90=lambda s: float(np.percentile(s, 90)),
            max="max",
        )
        self._table("chars per message, by role", stats.round(1))

    def _busiest_threads(self, df: pd.DataFrame, top: int) -> None:
        threads = (
            df.groupby(["thread", "title"], dropna=False)
            .agg(
                messages=("id", "count"),
                chars=("chars", "sum"),
                started=("created_at", "min"),
                ended=("created_at", "max"),
            )
            .sort_values("messages", ascending=False)
            .head(top)
        )
        threads["minutes"] = (
            (threads["ended"] - threads["started"]).dt.total_seconds() / 60
        ).round(1)
        self._table(
            f"busiest threads (top {top})",
            threads.drop(columns=["ended"]).assign(
                started=threads["started"].dt.strftime("%Y-%m-%d %H:%M")
            ),
        )

    def _activity(self, df: pd.DataFrame) -> None:
        daily = df.set_index("created_at")["id"].resample("D").count()
        self.stdout.write(f"\ndaily volume ({len(daily)} days)")
        self.stdout.write(f"  {sparkline(daily.to_numpy())}  peak {daily.max()}/day")

        hourly = df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        self.stdout.write("hour of day (UTC, 00→23)")
        self.stdout.write(f"  {sparkline(hourly.to_numpy())}  busiest {hourly.idxmax():02d}:00")

    def _table(self, title: str, frame: pd.DataFrame) -> None:
        self.stdout.write(f"\n{title}")
        self.stdout.write(frame.to_string())
