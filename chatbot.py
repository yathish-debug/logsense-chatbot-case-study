"""LogSense - AI-powered application log investigation chatbot.

Core logic lives here: log parsing, analytics (frequency, clustering, spike
detection, failure correlation), a retrieval layer over a troubleshooting
knowledge base, and an optional LLM explanation step.

Design rule enforced throughout: OBSERVED FACTS (with log line numbers) are
reported separately from POSSIBLE EXPLANATIONS. The bot never claims a root
cause is confirmed.

Run as a CLI:
    python chatbot.py --log data/sample_app.log
    python chatbot.py --log data/sample_app.log --ask "why did the app crash?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_KB_PATH = Path(__file__).parent / "data" / "troubleshooting_kb.json"

#: Canonical severity ordering, lowest to highest.
SEVERITY_ORDER = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: Raw level tokens found in the wild, mapped to canonical severities.
LEVEL_ALIASES = {
    "TRACE": "TRACE", "FINEST": "TRACE", "VERBOSE": "TRACE",
    "DEBUG": "DEBUG", "FINE": "DEBUG",
    "INFO": "INFO", "INFORMATION": "INFO", "NOTICE": "INFO",
    "WARN": "WARNING", "WARNING": "WARNING",
    "ERROR": "ERROR", "ERR": "ERROR", "SEVERE": "ERROR", "EXCEPTION": "ERROR",
    "CRITICAL": "CRITICAL", "CRIT": "CRITICAL", "FATAL": "CRITICAL",
    "ALERT": "CRITICAL", "EMERG": "CRITICAL", "PANIC": "CRITICAL",
}

PROBLEM_LEVELS = ("WARNING", "ERROR", "CRITICAL")
FAILURE_LEVELS = ("ERROR", "CRITICAL")

#: Keyword -> subsystem category. Used for category queries ("show me all
#: database errors") and as a correlation hint between failures.
SUBSYSTEM_KEYWORDS = {
    "database": ("database", "db", "postgres", "mysql", "sql", "jdbc", "mongo",
                 "redis", "connection pool", "deadlock", "psycopg", "sqlalchemy"),
    "network": ("connection refused", "connection reset", "dns", "socket",
                "unreachable", "tls", "ssl", "handshake", "econnre"),
    "http_api": ("http 4", "http 5", "status 4", "status 5", "upstream",
                 "gateway", "endpoint", "request failed", "500", "502", "503",
                 "429", "bad request"),
    "timeout": ("timeout", "timed out", "deadline exceeded", "read timeout"),
    "auth": ("auth", "unauthorized", "forbidden", "token", "credential",
             "permission denied", "401", "403"),
    "memory": ("outofmemory", "out of memory", "oom", "heap", "memoryerror",
               "cannot allocate"),
    "disk": ("no space left", "disk", "quota exceeded", "ioerror", "read-only"),
    "worker_queue": ("worker", "celery", "consumer", "scheduler", "cron",
                     "queue depth", "message queue", "background job",
                     "job failed", "broker"),
    "config": ("config", "environment variable", "missing key", "not set",
               "invalid setting"),
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LogRecord:
    """One logical log entry (a header line plus any continuation lines)."""

    line_no: int
    raw: str
    level: str = "UNKNOWN"
    message: str = ""
    timestamp: datetime | None = None
    continuations: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Header message plus stack trace / continuation lines."""
        return "\n".join([self.message, *self.continuations]).strip()

    @property
    def has_stack_trace(self) -> bool:
        blob = self.full_text.lower()
        return any(m in blob for m in ("traceback (most recent call last)",
                                       "\tat ", "\n  at ", "caused by:", '  file "'))

    @property
    def is_problem(self) -> bool:
        return self.level in PROBLEM_LEVELS

    @property
    def is_failure(self) -> bool:
        return self.level in FAILURE_LEVELS


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class LogParser:
    """Turns raw log text into :class:`LogRecord` objects.

    Handles plain-text logs (ISO-8601 or syslog timestamps, bracketed or bare
    severity tokens), JSON-lines logs, and multi-line stack traces. Unmatched
    indented lines are folded into the preceding record so a Java or Python
    traceback stays attached to its error.
    """

    # Timestamp formats, most specific first.
    TS_PATTERNS: Sequence[tuple[re.Pattern[str], Sequence[str]]] = (
        (re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)"),
         ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
          "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")),
        (re.compile(r"(\d{2}/[A-Za-z]{3}/\d{4}[: ]\d{2}:\d{2}:\d{2})"),
         ("%d/%b/%Y:%H:%M:%S", "%d/%b/%Y %H:%M:%S")),
        (re.compile(r"([A-Z][a-z]{2} +\d{1,2} \d{2}:\d{2}:\d{2})"),
         ("%b %d %H:%M:%S",)),
    )

    LEVEL_RE = re.compile(
        r"(?<![A-Za-z])(" + "|".join(sorted(LEVEL_ALIASES, key=len, reverse=True)) +
        r")(?![A-Za-z])", re.IGNORECASE)

    CONTINUATION_RE = re.compile(
        r"^(\s+|Traceback \(most recent call last\)|Caused by:|\.\.\.\s*\d+ more|"
        r"at [A-Za-z_$][\w.$]*\()")

    # Patterns used to turn a concrete message into a reusable signature.
    _NORMALIZERS: Sequence[tuple[re.Pattern[str], str]] = (
        (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                    re.I), "<UUID>"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
        (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<HEX>"),
        (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<HASH>"),
        (re.compile(r"(['\"])(?:(?!\1).){1,80}\1"), "<STR>"),
        (re.compile(r"(?:/[\w.\-]+){2,}/?"), "<PATH>"),
        (re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|kb|mb|gb)?\b", re.I), "<NUM>"),
    )

    EXCEPTION_RE = re.compile(
        r"\b([A-Z][A-Za-z0-9_]*(?:\.[A-Z][A-Za-z0-9_]*)*"
        r"(?:Error|Exception|Timeout|TimeoutError|Failure|Fault|Refused))\b")

    def parse_text(self, text: str) -> list[LogRecord]:
        return self.parse_lines(text.splitlines())

    def parse_lines(self, lines: Iterable[str]) -> list[LogRecord]:
        records: list[LogRecord] = []
        for idx, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            record = self._parse_json_line(idx, raw) or self._parse_text_line(idx, raw)
            if record is None:
                # Continuation of the previous entry (stack trace frame, etc.).
                if records:
                    records[-1].continuations.append(raw.rstrip())
                else:
                    records.append(LogRecord(
                                    line_no=idx,
                                    raw=raw,
                                    level=self._infer_level(raw),
                                    message=raw.strip(),
                            ))
                continue
            records.append(record)
        self._backfill_timestamps(records)
        return records

    # -- line-level parsing -------------------------------------------------

    def _parse_json_line(self, line_no: int, raw: str) -> LogRecord | None:
        stripped = raw.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return None
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        level = "UNKNOWN"
        for key in ("level", "severity", "levelname", "log.level", "loglevel"):
            if key in payload:
                level = LEVEL_ALIASES.get(str(payload[key]).upper(), "UNKNOWN")
                break

        message = ""
        for key in ("message", "msg", "body", "event", "log"):
            if key in payload:
                message = str(payload[key])
                break

        timestamp = None
        for key in ("timestamp", "time", "ts", "@timestamp", "datetime"):
            if key in payload:
                timestamp = self._parse_timestamp(str(payload[key]))
                break

        extras = [f"{k}={v}" for k, v in payload.items()
                  if k not in {"level", "severity", "levelname", "log.level",
                               "loglevel", "message", "msg", "body", "event",
                               "log", "timestamp", "time", "ts", "@timestamp",
                               "datetime"}]
        record = LogRecord(line_no=line_no, raw=raw.rstrip(), level=level,
                           message=message or stripped, timestamp=timestamp)
        if extras:
            record.continuations.append(" ".join(extras))
        if record.level == "UNKNOWN":
            record.level = self._infer_level(record.full_text)
        return record

    def _parse_text_line(self, line_no: int, raw: str) -> LogRecord | None:
        if self.CONTINUATION_RE.match(raw):
            return None

        timestamp, ts_span = self._extract_timestamp(raw)
        level_match = self.LEVEL_RE.search(raw)
        level = (LEVEL_ALIASES.get(level_match.group(1).upper(), "UNKNOWN")
                 if level_match else "UNKNOWN")

        # Treat a line with neither timestamp nor level as a continuation,
        # but only if we already have something to attach it to.
        if timestamp is None and level_match is None:
            return None

        # Message = everything after the timestamp and level tokens.
        cut = 0
        if ts_span:
            cut = max(cut, ts_span[1])
        if level_match and level_match.end() > cut:
            cut = level_match.end()
        message = self._strip_prefix(raw[cut:])

        level = level or "UNKNOWN"
        record = LogRecord(line_no=line_no, raw=raw.rstrip(), level=level,
                           message=message or raw.strip(), timestamp=timestamp)
        if record.level == "UNKNOWN":
            record.level = self._infer_level(record.full_text)
        return record

    _PREFIX_JUNK_RE = re.compile(r"^[\s\-:|\]]+")
    _THREAD_RE = re.compile(r"^\[[^\]]{1,48}\]\s*")

    @classmethod
    def _strip_prefix(cls, remainder: str) -> str:
        """Drop leftover separators and the thread/context bracket.

        ``] [http-4] db.session - Timeout`` becomes ``db.session - Timeout``,
        so thread names never end up inside an error signature.
        """
        text = cls._PREFIX_JUNK_RE.sub("", remainder)
        text = cls._THREAD_RE.sub("", text)
        return cls._PREFIX_JUNK_RE.sub("", text).strip()

    def _extract_timestamp(self, raw: str) -> tuple[datetime | None, tuple[int, int] | None]:
        for pattern, formats in self.TS_PATTERNS:
            match = pattern.search(raw)
            if not match:
                continue
            parsed = self._try_formats(match.group(1), formats)
            if parsed is not None:
                return parsed, match.span(1)
        return None, None

    def _parse_timestamp(self, value: str) -> datetime | None:
        value = value.strip().rstrip("Z")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
        for pattern, formats in self.TS_PATTERNS:
            match = pattern.search(value)
            if match:
                parsed = self._try_formats(match.group(1), formats)
                if parsed is not None:
                    return parsed
        return None

    @staticmethod
    def _try_formats(value: str, formats: Sequence[str]) -> datetime | None:
        candidate = value.replace(",", ".")
        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            # Syslog format carries no year; assume the current one.
            if "%Y" not in fmt:
                parsed = parsed.replace(year=datetime.now().year)
            return parsed
        return None

    @staticmethod
    def _infer_level(text: str) -> str:
        """Last-resort severity guess for lines with no explicit level token."""
        lowered = text.lower()
        if any(k in lowered for k in ("traceback", "exception", "fatal", "panic")):
            return "ERROR"
        if "fail" in lowered or "error" in lowered:
            return "ERROR"
        if "warn" in lowered or "deprecat" in lowered:
            return "WARNING"
        return "INFO"

    @staticmethod
    def _backfill_timestamps(records: list[LogRecord]) -> None:
        """Give undated records the timestamp of the nearest dated one above."""
        last: datetime | None = None
        for record in records:
            if record.timestamp is not None:
                last = record.timestamp
            elif last is not None:
                record.timestamp = last

    # -- signature helpers --------------------------------------------------

    @classmethod
    def signature(cls, message: str) -> str:
        """Collapse a concrete message into a template for grouping.

        ``Timeout connecting to db-7 after 3000ms`` and ``Timeout connecting to
        db-2 after 5000ms`` both reduce to the same signature, so repeated
        errors can be counted together.
        """
        text = message.split("\n", 1)[0].strip()
        for pattern, placeholder in cls._NORMALIZERS:
            text = pattern.sub(placeholder, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:200] or "<empty>"

    @classmethod
    def exception_names(cls, text: str) -> list[str]:
        seen: dict[str, None] = {}
        for name in cls.EXCEPTION_RE.findall(text):
            seen.setdefault(name.split(".")[-1], None)
        return list(seen)


def _compile_keyword(keyword: str) -> re.Pattern[str]:
    """Word-bounded pattern for single words, plain substring for phrases.

    Without the word boundary, short keywords produce false positives: a
    ``QueuePool`` timeout would otherwise be tagged as a queue problem.
    """
    if re.fullmatch(r"[a-z0-9]+", keyword):
        return re.compile(rf"\b{re.escape(keyword)}\b")
    return re.compile(re.escape(keyword))


_SUBSYSTEM_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    name: tuple(_compile_keyword(k) for k in keywords)
    for name, keywords in SUBSYSTEM_KEYWORDS.items()
}


def classify_subsystems(text: str) -> list[str]:
    """Tag a message with the subsystems it mentions (database, auth, ...)."""
    lowered = text.lower()
    return [name for name, patterns in _SUBSYSTEM_PATTERNS.items()
            if any(p.search(lowered) for p in patterns)]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@dataclass
class ErrorGroup:
    """A set of log records sharing one normalized signature."""

    signature: str
    count: int
    level: str
    line_numbers: list[int]
    sample: str
    exceptions: list[str]
    subsystems: list[str]
    first_seen: datetime | None
    last_seen: datetime | None

    def cite(self, limit: int = 3) -> str:
        shown = ", ".join(f"L{n}" for n in self.line_numbers[:limit])
        extra = len(self.line_numbers) - limit
        return f"{shown}{f' (+{extra} more)' if extra > 0 else ''}"


class LogAnalytics:
    """Deterministic analysis over parsed records.

    Everything here is computed from the log itself, so the figures the
    chatbot quotes are verifiable rather than generated.
    """

    def __init__(self, records: Sequence[LogRecord]):
        self.records = list(records)
        self.groups = self._build_groups()

    # -- basics -------------------------------------------------------------

    @property
    def level_counts(self) -> dict[str, int]:
        counts = Counter(r.level for r in self.records)
        ordered = {lvl: counts.get(lvl, 0) for lvl in SEVERITY_ORDER
                   if counts.get(lvl, 0)}
        if counts.get("UNKNOWN"):
            ordered["UNKNOWN"] = counts["UNKNOWN"]
        return ordered

    @property
    def time_span(self) -> tuple[datetime | None, datetime | None]:
        stamps = [r.timestamp for r in self.records if r.timestamp]
        return (min(stamps), max(stamps)) if stamps else (None, None)

    @property
    def failures(self) -> list[LogRecord]:
        return [r for r in self.records if r.is_failure]

    @property
    def first_failure(self) -> LogRecord | None:
        return self.failures[0] if self.failures else None

    # -- grouping -----------------------------------------------------------

    def _build_groups(self) -> list[ErrorGroup]:
        buckets: dict[tuple[str, str], list[LogRecord]] = defaultdict(list)
        for record in self.records:
            if record.is_problem:
                buckets[(record.level, LogParser.signature(record.message))].append(record)

        groups: list[ErrorGroup] = []
        for (level, signature), members in buckets.items():
            blob = "\n".join(m.full_text for m in members)
            stamps = [m.timestamp for m in members if m.timestamp]
            groups.append(ErrorGroup(
                signature=signature,
                count=len(members),
                level=level,
                line_numbers=[m.line_no for m in members],
                sample=members[0].message.split("\n", 1)[0][:300],
                exceptions=LogParser.exception_names(blob),
                subsystems=classify_subsystems(blob),
                first_seen=min(stamps) if stamps else None,
                last_seen=max(stamps) if stamps else None,
            ))
        groups.sort(key=lambda g: (-g.count, SEVERITY_ORDER.index(g.level)
                                   if g.level in SEVERITY_ORDER else 0))
        return groups

    def top_groups(self, limit: int = 5, levels: Sequence[str] = FAILURE_LEVELS
                   ) -> list[ErrorGroup]:
        return [g for g in self.groups if g.level in levels][:limit]

    def groups_for_subsystem(self, subsystem: str) -> list[ErrorGroup]:
        return [g for g in self.groups if subsystem in g.subsystems]

    def recurring_exceptions(self, min_count: int = 2) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for group in self.groups:
            for name in group.exceptions:
                counter[name] += group.count
        return [(name, n) for name, n in counter.most_common() if n >= min_count]

    # -- clustering ---------------------------------------------------------

    def cluster_groups(self, min_groups: int = 4, eps: float = 0.45
                       ) -> list[list[ErrorGroup]]:
        """Group *similar* error signatures that differ in wording.

        Uses TF-IDF character n-grams plus DBSCAN on cosine distance, so
        ``Could not reach payment service`` and ``Payment service
        unreachable`` can land in one cluster. Returns an empty list when
        scikit-learn is unavailable or there is too little to cluster.
        """
        candidates = [g for g in self.groups if g.level in FAILURE_LEVELS]
        if len(candidates) < min_groups:
            return []
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return []

        corpus = [f"{g.signature} {' '.join(g.exceptions)}" for g in candidates]
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=1).fit_transform(corpus)
        labels = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(matrix)

        clustered: dict[int, list[ErrorGroup]] = defaultdict(list)
        for group, label in zip(candidates, labels):
            if label != -1:
                clustered[int(label)].append(group)
        return sorted(clustered.values(),
                      key=lambda members: -sum(m.count for m in members))

    # -- timeline and anomalies --------------------------------------------

    def failure_timeline(self, bucket_seconds: int = 60
                         ) -> list[tuple[datetime, int]]:
        buckets: Counter[datetime] = Counter()
        for record in self.failures:
            if record.timestamp is None:
                continue
            epoch = int(record.timestamp.timestamp())
            floored = epoch - (epoch % bucket_seconds)
            buckets[datetime.fromtimestamp(floored)] += 1
        return sorted(buckets.items())

    def detect_spikes(self, bucket_seconds: int = 60, z_threshold: float = 2.0
                      ) -> list[tuple[datetime, int, float]]:
        """Flag time buckets whose failure count is unusually high.

        Reported as ``(bucket_start, count, z_score)``. With fewer than three
        populated buckets there is no distribution to compare against, so
        nothing is flagged.
        """
        timeline = self.failure_timeline(bucket_seconds)
        counts = [count for _, count in timeline]
        if len(counts) < 3:
            return []
        mean = statistics.fmean(counts)
        stdev = statistics.pstdev(counts)
        if stdev == 0:
            return []
        spikes = [(start, count, (count - mean) / stdev)
                  for start, count in timeline
                  if (count - mean) / stdev >= z_threshold and count >= 3]
        return sorted(spikes, key=lambda item: -item[2])

    def events_before(self, record: LogRecord, window_seconds: int = 120,
                      limit: int = 8) -> list[LogRecord]:
        """Problem records shortly before ``record`` - correlation candidates."""
        preceding = [r for r in self.records
                     if r.line_no < record.line_no and r.is_problem]
        if record.timestamp is not None:
            cutoff = record.timestamp - timedelta(seconds=window_seconds)
            dated = [r for r in preceding
                     if r.timestamp is not None and r.timestamp >= cutoff]
            if dated:
                preceding = dated
        return preceding[-limit:]

    def correlate_failures(self) -> dict:
        """Build an evidence bundle for "why did this break?" questions.

        Returns the first failure, what preceded it, which error signature
        appeared earliest, and whether the earliest and the loudest failures
        share a subsystem - the signal that they may be one incident rather
        than several.
        """
        first = self.first_failure
        if first is None:
            return {}

        dated = [g for g in self.groups
                 if g.level in FAILURE_LEVELS and g.first_seen is not None]
        earliest = min(dated, key=lambda g: g.first_seen) if dated else None
        loudest = next((g for g in self.groups if g.level in FAILURE_LEVELS), None)

        shared: list[str] = []
        if earliest and loudest and earliest is not loudest:
            shared = sorted(set(earliest.subsystems) & set(loudest.subsystems))

        return {
            "first_failure": first,
            "preceding": self.events_before(first),
            "earliest_group": earliest,
            "loudest_group": loudest,
            "shared_subsystems": shared,
            "spikes": self.detect_spikes(),
        }


# ---------------------------------------------------------------------------
# Retrieval-augmented generation layer
# ---------------------------------------------------------------------------


@dataclass
class KBEntry:
    entry_id: str
    title: str
    subsystem: str
    symptoms: list[str]
    likely_causes: list[str]
    checks: list[str]
    keywords: list[str]

    @property
    def search_text(self) -> str:
        return " ".join([self.title, self.subsystem, *self.symptoms,
                         *self.likely_causes, *self.keywords])


def load_knowledge_base(path: str | Path = DEFAULT_KB_PATH) -> list[KBEntry]:
    """Load the troubleshooting datafile used for retrieval.

    Accepts the project's JSON format (``{"entries": [...]}`` or a bare list).
    Returns an empty list if the file is missing, so the bot still runs.
    """
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else payload

    entries: list[KBEntry] = []
    for raw in raw_entries:
        entries.append(KBEntry(
            entry_id=str(raw.get("id", f"kb-{len(entries) + 1}")),
            title=str(raw.get("title", "Untitled")),
            subsystem=str(raw.get("subsystem", "general")),
            symptoms=list(raw.get("symptoms", [])),
            likely_causes=list(raw.get("likely_causes", [])),
            checks=list(raw.get("checks", [])),
            keywords=list(raw.get("keywords", [])),
        ))
    return entries


class KnowledgeBase:
    """TF-IDF retrieval over troubleshooting entries.

    A vector database is overkill for a few dozen curated entries; TF-IDF
    cosine similarity keeps the dependency list short and the results
    inspectable. Falls back to keyword overlap scoring without scikit-learn.
    """

    def __init__(self, entries: Sequence[KBEntry]):
        self.entries = list(entries)
        self._vectorizer = None
        self._matrix = None
        if self.entries:
            self._fit()

    def _fit(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self._matrix = self._vectorizer.fit_transform(
            [e.search_text for e in self.entries])

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.05
                 ) -> list[tuple[KBEntry, float]]:
        if not self.entries or not query.strip():
            return []
        if self._vectorizer is None or self._matrix is None:
            return self._keyword_retrieve(query, k)

        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(self._vectorizer.transform([query]),
                                   self._matrix)[0]
        ranked = sorted(zip(self.entries, scores), key=lambda p: -p[1])
        hits = [(entry, float(score)) for entry, score in ranked[:k]
                if score >= min_score]
        return hits or self._keyword_retrieve(query, k)

    def _keyword_retrieve(self, query: str, k: int) -> list[tuple[KBEntry, float]]:
        tokens = set(re.findall(r"[a-z]{3,}", query.lower()))
        scored = []
        for entry in self.entries:
            words = set(re.findall(r"[a-z]{3,}", entry.search_text.lower()))
            overlap = len(tokens & words)
            if overlap:
                scored.append((entry, overlap / max(len(tokens), 1)))
        return sorted(scored, key=lambda p: -p[1])[:k]


# ---------------------------------------------------------------------------
# Optional LLM layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are LogSense, a log investigation assistant for developers.

Rules you must follow:
1. Use only the EVIDENCE provided. Never invent log lines, counts or timestamps.
2. Separate what the log shows from what it might mean. Label the two sections
   "Observed" and "Possible explanations".
3. Cite line numbers (L<number>) for every observed fact.
4. Hedge causal claims ("may indicate", "is consistent with"). Never state a
   root cause as confirmed.
5. If the evidence does not answer the question, say so plainly.
6. Be concise. Finish with up to four concrete checks the developer can run."""


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat endpoint.

    Works with the OpenAI API, or any compatible server (Ollama, vLLM,
    OpenRouter) via ``OPENAI_BASE_URL``. When no key is configured the
    chatbot runs in offline mode and answers from analytics plus the
    knowledge base alone.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model or os.getenv("LOGSENSE_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None

    @property
    def available(self) -> bool:
        if not self.api_key and not self.base_url:
            return False
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                return False
            kwargs = {"api_key": self.api_key or "not-needed"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            try:
                self._client = OpenAI(**kwargs)
            except Exception:
                return False
        return True

    def explain(self, question: str, evidence: str) -> str | None:
        if not self.available:
            return None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}"},
                ],
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception as exc:  # network, auth, quota - degrade quietly
            return f"_LLM unavailable ({type(exc).__name__}); showing analysis only._"


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

#: Question keyword -> intent. Checked in order, first match wins.
INTENT_RULES: Sequence[tuple[str, tuple[str, ...]]] = (
    ("root_cause", ("root cause", "why did", "what caused", "crash", "went wrong",
                    "investigate first", "break", "broke")),
    ("timeline", ("before", "timeline", "sequence", "leading up", "order of",
                  "preceded", "preceding")),
    ("summary", ("summar", "overview", "what happened", "tldr", "tl;dr")),
    ("top_errors", ("most frequent", "most common", "top error", "frequent error",
                    "biggest", "worst")),
    ("spikes", ("spike", "anomal", "unusual", "burst", "sudden")),
    ("recurring", ("recurring", "repeat", "again and again", "same error")),
    ("clusters", ("similar", "cluster", "group", "related error")),
)


class LogSense:
    """The chatbot: load a log, then ask questions about it."""

    def __init__(self, kb_path: str | Path = DEFAULT_KB_PATH,
                 llm: LLMClient | None = None):
        self.parser = LogParser()
        self.kb = KnowledgeBase(load_knowledge_base(kb_path))
        self.llm = llm if llm is not None else LLMClient()
        self.records: list[LogRecord] = []
        self.analytics: LogAnalytics | None = None
        self.source = "<none>"

    # -- loading ------------------------------------------------------------

    def load_text(self, text: str, source: str = "<uploaded>") -> LogAnalytics:
        self.records = self.parser.parse_text(text)
        self.analytics = LogAnalytics(self.records)
        self.source = source
        return self.analytics

    def load_file(self, path: str | Path) -> LogAnalytics:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.load_text(text, source=path.name)

    def _require_log(self) -> LogAnalytics:
        if self.analytics is None:
            raise RuntimeError("No log loaded. Call load_file() or load_text() first.")
        return self.analytics

    # -- reports ------------------------------------------------------------

    def summary(self) -> str:
        analytics = self._require_log()
        start, end = analytics.time_span
        counts = analytics.level_counts
        lines = [
            f"**Log summary - {self.source}**",
            f"- Entries parsed: {len(analytics.records)}",
            "- Severity: " + (", ".join(f"{lvl} {n}" for lvl, n in counts.items())
                              or "none detected"),
        ]
        if start and end:
            lines.append(f"- Window: {start:%Y-%m-%d %H:%M:%S} to {end:%H:%M:%S}")
        traces = sum(1 for r in analytics.records if r.has_stack_trace)
        if traces:
            lines.append(f"- Entries with stack traces: {traces}")

        top = analytics.top_groups(5)
        if top:
            lines.append("\n**Most frequent failures**")
            for i, group in enumerate(top, 1):
                tag = f" [{', '.join(group.subsystems)}]" if group.subsystems else ""
                lines.append(f"{i}. ({group.count}x) {group.sample}{tag} - {group.cite()}")
        else:
            lines.append("\nNo ERROR or CRITICAL entries found.")

        spikes = analytics.detect_spikes()
        if spikes:
            start_at, count, z = spikes[0]
            lines.append(f"\n**Unusual activity**: {count} failures around "
                         f"{start_at:%H:%M} (z={z:.1f}).")
        return "\n".join(lines)

    def root_cause_report(self) -> str:
        analytics = self._require_log()
        evidence = analytics.correlate_failures()
        if not evidence:
            return "No ERROR or CRITICAL entries in this log, so there is no failure to trace."

        first: LogRecord = evidence["first_failure"]
        earliest: ErrorGroup | None = evidence["earliest_group"]
        loudest: ErrorGroup | None = evidence["loudest_group"]

        observed = [
            f"- First failure at L{first.line_no}"
            + (f" ({first.timestamp:%H:%M:%S})" if first.timestamp else "")
            + f": {first.message.split(chr(10))[0][:200]}"
        ]

        if first.has_stack_trace:
            trace_lines = first.full_text.splitlines()
            exception_line = next(
                (
                    line.strip()
                    for line in reversed(trace_lines)
                    if LogParser.EXCEPTION_RE.search(line)
                ),
                None,
            )
            python_frames = [
                line.strip()
                for line in trace_lines
                if 'File "' in line and ", line " in line
            ]

            if exception_line:
                observed.append(
                    f"- Exception details at L{first.line_no}: {exception_line}"
                )
            if python_frames:
                observed.append(
                    f"- Last Python frame: {python_frames[-1]}"
                )

        for record in evidence["preceding"]:
            observed.append(
                f"- Before it, L{record.line_no} [{record.level}]: "
                f"{record.message.split(chr(10))[0][:160]}"
            )

        if loudest:
            observed.append(
                f"- Highest-volume failure ({loudest.count}x): "
                f"{loudest.sample} - {loudest.cite()}"
            )

        if earliest and earliest is not loudest:
            observed.append(
                f"- Earliest distinct failure signature: "
                f"{earliest.sample} - {earliest.cite()}"
            )

        hypotheses = []
        if earliest and loudest and earliest is not loudest:
            shared = evidence["shared_subsystems"]
            link = (
                f" Both touch the {', '.join(shared)} subsystem, "
                "which may mean they are one incident rather than two."
                if shared
                else ""
            )
            hypotheses.append(
                f"- The earlier failure ({earliest.sample[:120]}) may have "
                f"triggered the later ones; it appears first in the log.{link}"
            )

        if first.has_stack_trace:
            hypotheses.append(
                "- The first failure carries a stack trace, so the "
                "exception origin is visible in the log itself."
            )

        for subsystem in dict.fromkeys(
            s for group in analytics.top_groups(3) for s in group.subsystems
        ):
            hypotheses.append(
                f"- Failures reference the {subsystem} subsystem; "
                f"a {subsystem} fault is consistent with this pattern."
            )

        if not hypotheses:
            hypotheses.append(
                "- The log does not show a clear ordering between failures, "
                "so no causal chain can be proposed."
            )

        checks = self._checks_for(analytics)
        report = [
            "**Observed (from the log)**",
            *observed,
            "\n**Possible explanations (not confirmed)**",
            *hypotheses[:4],
        ]
        if checks:
            report += ["\n**Suggested checks**", *checks]

        return "\n".join(report)

    def _checks_for(self, analytics: LogAnalytics, k: int = 2) -> list[str]:
        """Pull investigation steps from the knowledge base for the top errors."""
        query = " ".join(f"{g.signature} {' '.join(g.exceptions)} "
                         f"{' '.join(g.subsystems)}"
                         for g in analytics.top_groups(3))
        checks: list[str] = []
        for entry, score in self.kb.retrieve(query, k=k):
            for check in entry.checks[:2]:
                checks.append(f"- {check}  _[{entry.entry_id}: {entry.title}]_")
        return checks

    # -- question answering -------------------------------------------------

    def ask(self, question: str, use_llm: bool = True) -> str:
        analytics = self._require_log()
        intent = self._classify(question)
        grounded = self._grounded_answer(intent, question, analytics)

        if not use_llm or not self.llm.available:
            return grounded

        kb_context = "\n".join(
            f"[{entry.entry_id}] {entry.title} | causes: "
            f"{'; '.join(entry.likely_causes[:3])} | checks: "
            f"{'; '.join(entry.checks[:3])}"
            for entry, _ in self.kb.retrieve(f"{question} {grounded[:800]}", k=3))

        evidence = (f"LOG ANALYSIS (computed, trustworthy):\n{grounded}\n\n"
                    f"TROUBLESHOOTING KNOWLEDGE BASE:\n{kb_context or 'none'}")
        answer = self.llm.explain(question, evidence)
        return answer or grounded

    @staticmethod
    def _classify(question: str) -> str:
        """Route a question to a handler.

        Explicit analysis intents win over subsystem filters, so "why did the
        database fail?" gets a root-cause report rather than a flat list of
        database lines.
        """
        lowered = question.lower()
        for intent, keywords in INTENT_RULES:
            if any(keyword in lowered for keyword in keywords):
                return intent
        for subsystem in classify_subsystems(lowered):
            return f"subsystem:{subsystem}"
        return "search"

    def _grounded_answer(self, intent: str, question: str,
                         analytics: LogAnalytics) -> str:
        if intent == "summary":
            return self.summary()
        if intent == "root_cause":
            return self.root_cause_report()

        if intent == "top_errors":
            top = analytics.top_groups(5)
            if not top:
                return "No ERROR or CRITICAL entries found in this log."
            return "**Most frequent failures**\n" + "\n".join(
                f"{i}. ({g.count}x) {g.sample} - {g.cite()}"
                for i, g in enumerate(top, 1))

        if intent == "timeline":
            first = analytics.first_failure
            if first is None:
                return "No failures found, so there is nothing to build a timeline around."
            before = analytics.events_before(first)
            body = "\n".join(f"- L{r.line_no} [{r.level}] "
                             f"{r.message.split(chr(10))[0][:160]}" for r in before)
            return (f"**Events before the first failure (L{first.line_no})**\n"
                    f"{body or '- No warnings or errors precede it.'}\n\n"
                    f"First failure: {first.message.split(chr(10))[0][:200]}")

        if intent == "spikes":
            spikes = analytics.detect_spikes()
            if not spikes:
                return "No statistically unusual failure bursts detected."
            return "**Unusual failure bursts**\n" + "\n".join(
                f"- {start:%Y-%m-%d %H:%M} - {count} failures (z={z:.1f})"
                for start, count, z in spikes[:5])

        if intent == "recurring":
            recurring = analytics.recurring_exceptions()
            repeated = [g for g in analytics.groups
                        if g.count > 1 and g.level in FAILURE_LEVELS][:5]
            parts = []
            if recurring:
                parts.append("**Recurring exceptions**\n" + "\n".join(
                    f"- {name} x{n}" for name, n in recurring[:5]))
            if repeated:
                parts.append("**Repeated error signatures**\n" + "\n".join(
                    f"- ({g.count}x) {g.sample} - {g.cite()}" for g in repeated))
            return "\n\n".join(parts) or "No error repeats more than once."

        if intent == "clusters":
            clusters = analytics.cluster_groups()
            if not clusters:
                distinct = len([g for g in analytics.groups
                                if g.level in FAILURE_LEVELS])
                if distinct < 4:
                    return (f"Only {distinct} distinct failure signature(s) in this "
                            f"log - too few to cluster meaningfully.")
                return (f"Compared {distinct} distinct failure signatures, but none "
                        f"were similar enough to group. They look like separate "
                        f"problems rather than variants of one.")
            out = ["**Clusters of similar errors**"]
            for i, members in enumerate(clusters, 1):
                total = sum(m.count for m in members)
                out.append(f"\nCluster {i} ({total} occurrences):")
                out += [f"  - ({m.count}x) {m.sample[:140]} - {m.cite(2)}"
                        for m in members]
            return "\n".join(out)

        if intent.startswith("subsystem:"):
            subsystem = intent.split(":", 1)[1]
            groups = analytics.groups_for_subsystem(subsystem)
            if not groups:
                return f"No {subsystem.replace('_', ' ')} entries found in this log."
            return (f"**{subsystem.replace('_', ' ').title()} entries**\n" +
                    "\n".join(f"- [{g.level}] ({g.count}x) {g.sample} - {g.cite()}"
                              for g in groups[:8]))

        # Fallback: keyword search across the log, plus KB context.
        return self._search(question, analytics)

    def _search(self, question: str, analytics: LogAnalytics) -> str:
        tokens = [t for t in re.findall(r"[A-Za-z0-9_]{3,}", question.lower())
                  if t not in {"the", "log", "logs", "what", "show", "any", "all",
                               "and", "are", "was", "were", "about", "this", "file",
                               "there", "which", "does", "did", "for", "with"}]
        hits = [r for r in analytics.records
                if any(t in r.full_text.lower() for t in tokens)][:8] if tokens else []

        parts = []
        if hits:
            parts.append("**Matching log entries**\n" + "\n".join(
                f"- L{r.line_no} [{r.level}] {r.message.split(chr(10))[0][:180]}"
                for r in hits))
        else:
            parts.append("No log entries match that wording. Summary instead:\n\n"
                         + self.summary())

        kb_hits = self.kb.retrieve(question, k=2)
        if kb_hits:
            parts.append("**Related troubleshooting guidance**\n" + "\n".join(
                f"- [{e.entry_id}] {e.title}: {'; '.join(e.checks[:2])}"
                for e, _ in kb_hits))
        return "\n\n".join(parts)

    # -- advanced -----------------------------------------------------------

    def compare_with(self, other_path: str | Path) -> str:
        """Diff the error signatures of the loaded log against another file."""
        current = self._require_log()
        other = LogAnalytics(self.parser.parse_text(
            Path(other_path).read_text(encoding="utf-8", errors="replace")))

        def sigs(a: LogAnalytics) -> dict[str, ErrorGroup]:
            return {g.signature: g for g in a.groups if g.level in FAILURE_LEVELS}

        mine, theirs = sigs(current), sigs(other)
        new = [mine[s] for s in mine.keys() - theirs.keys()]
        gone = [theirs[s] for s in theirs.keys() - mine.keys()]
        lines = [f"**Comparison: {self.source} vs {Path(other_path).name}**",
                 f"- Failures: {len(current.failures)} vs {len(other.failures)}"]
        if new:
            lines.append("\n**Only in the loaded log**")
            lines += [f"- ({g.count}x) {g.sample} - {g.cite()}" for g in new[:5]]
        if gone:
            lines.append("\n**Only in the other log**")
            lines += [f"- ({g.count}x) {g.sample}" for g in gone[:5]]
        if not new and not gone:
            lines.append("\nBoth logs show the same error signatures.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BANNER = """LogSense - application log investigation chatbot
Type a question, or one of: /summary  /errors  /timeline  /spikes  /clusters  /quit
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatbot.py", description="Investigate an application log file.")
    parser.add_argument("--log", "-l", required=True, help="Path to a .log/.txt/.csv file")
    parser.add_argument("--ask", "-a", help="Ask one question and exit")
    parser.add_argument("--compare", "-c", help="Compare against a second log file")
    parser.add_argument("--kb", default=str(DEFAULT_KB_PATH),
                        help="Path to the troubleshooting knowledge base JSON")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM step and use computed analysis only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    bot = LogSense(kb_path=args.kb)
    try:
        bot.load_file(args.log)
    except FileNotFoundError:
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 1

    use_llm = not args.no_llm

    if args.compare:
        print(bot.compare_with(args.compare))
        return 0
    if args.ask:
        print(bot.ask(args.ask, use_llm=use_llm))
        return 0

    print(BANNER)
    print(bot.summary())
    if not bot.llm.available and use_llm:
        print("\n_No LLM configured (set OPENAI_API_KEY) - running on computed "
              "analysis and the knowledge base._")

    commands = {
        "/summary": bot.summary,
        "/errors": lambda: bot.ask("most frequent errors", use_llm=False),
        "/timeline": lambda: bot.ask("what happened before the first failure",
                                     use_llm=False),
        "/spikes": lambda: bot.ask("any unusual spikes", use_llm=False),
        "/clusters": lambda: bot.ask("cluster similar errors", use_llm=False),
    }

    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in {"/quit", "/exit", "quit", "exit"}:
            break
        handler = commands.get(question)
        print("\n" + (handler() if handler else bot.ask(question, use_llm=use_llm)))

    print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
