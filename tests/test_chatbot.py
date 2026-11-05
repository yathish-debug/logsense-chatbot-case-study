"""Tests for the LogSense core. Run with: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import (  # noqa: E402
    KnowledgeBase,
    LogAnalytics,
    LogParser,
    LogSense,
    classify_subsystems,
    load_knowledge_base,
)

SAMPLE_LOG = Path(__file__).resolve().parents[1] / "data" / "sample_app.log"


@pytest.fixture(scope="module")
def analytics() -> LogAnalytics:
    records = LogParser().parse_text(SAMPLE_LOG.read_text(encoding="utf-8"))
    return LogAnalytics(records)


# -- parsing ---------------------------------------------------------------


def test_parses_levels_and_timestamps(analytics: LogAnalytics):
    counts = analytics.level_counts
    assert counts["ERROR"] > 0
    assert counts["INFO"] > 0
    start, end = analytics.time_span
    assert start is not None and end is not None and start < end


def test_stack_trace_folds_into_preceding_record():
    text = (
        "2026-01-01 10:00:00 ERROR db - boom\n"
        'Traceback (most recent call last):\n'
        '  File "/srv/app.py", line 3, in run\n'
        "ValueError: bad input\n"
    )
    records = LogParser().parse_text(text)
    assert len(records) == 1, "trace lines must not become separate records"
    assert records[0].has_stack_trace
    assert "ValueError" in records[0].full_text

def test_standalone_traceback_is_classified_as_error():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/repro.py", line 1, in <module>\n'
        "TypeError: startswith first arg must be bytes or a tuple of bytes, not str\n"
    )

    (record,) = LogParser().parse_text(text)

    assert record.level == "ERROR"
    assert record.has_stack_trace
    assert "TypeError: startswith" in record.full_text
def test_parses_json_lines():
    text = '{"timestamp": "2026-01-01T10:00:00", "level": "ERROR", "message": "disk full"}'
    (record,) = LogParser().parse_text(text)
    assert record.level == "ERROR"
    assert record.message == "disk full"
    assert record.timestamp is not None


def test_thread_name_is_stripped_from_message():
    text = "2026-01-01 10:00:00,123 ERROR [http-4] db.session - Timeout after 3000ms"
    (record,) = LogParser().parse_text(text)
    assert record.message.startswith("db.session")
    assert "http-4" not in record.message


def test_level_aliases_normalize():
    text = ("2026-01-01 10:00:00 WARN  a - one\n"
            "2026-01-01 10:00:01 FATAL b - two\n"
            "2026-01-01 10:00:02 SEVERE c - three\n")
    levels = [r.level for r in LogParser().parse_text(text)]
    assert levels == ["WARNING", "CRITICAL", "ERROR"]


# -- signatures and grouping ----------------------------------------------


def test_signature_collapses_variable_parts():
    sig = LogParser.signature
    assert sig("Timeout connecting to db-1 after 3000ms") == \
           sig("Timeout connecting to db-1 after 9000ms")
    assert sig("Failed for user 'alice'") == sig("Failed for user 'bob'")


def test_repeated_errors_group_together(analytics: LogAnalytics):
    top = analytics.top_groups(1)[0]
    assert top.count > 1
    assert len(top.line_numbers) == top.count


def test_exception_names_extracted(analytics: LogAnalytics):
    names = {name for name, _ in analytics.recurring_exceptions(min_count=1)}
    assert "TimeoutError" in names
    assert "AttributeError" in names


def test_subsystem_classification_avoids_false_positives():
    # "QueuePool" must not be read as a message-queue problem.
    assert "worker_queue" not in classify_subsystems(
        "sqlalchemy.exc.TimeoutError: QueuePool limit of size 10 reached")
    assert "database" in classify_subsystems("db.session - connection timeout")
    assert "memory" in classify_subsystems("java.lang.OutOfMemoryError: heap space")


# -- analytics -------------------------------------------------------------


def test_first_failure_precedes_all_others(analytics: LogAnalytics):
    first = analytics.first_failure
    assert first is not None
    assert all(first.line_no <= r.line_no for r in analytics.failures)


def test_events_before_returns_only_earlier_problems(analytics: LogAnalytics):
    first = analytics.first_failure
    before = analytics.events_before(first)
    assert before, "sample log has warnings before the first failure"
    assert all(r.line_no < first.line_no and r.is_problem for r in before)


def test_spike_detection_flags_the_burst(analytics: LogAnalytics):
    spikes = analytics.detect_spikes()
    assert spikes, "sample log contains a deliberate error burst"
    _, count, z = spikes[0]
    assert count >= 3 and z >= 2.0


def test_spike_detection_quiet_on_flat_logs():
    text = "\n".join(
        f"2026-01-01 10:{m:02d}:00 ERROR svc - steady failure {m}" for m in range(10))
    assert LogAnalytics(LogParser().parse_text(text)).detect_spikes() == []


def test_clustering_groups_reworded_variants():
    text = "\n".join([
        "2026-01-01 10:00:00 ERROR a - payment service unreachable from gateway",
        "2026-01-01 10:00:01 ERROR b - payment service is unreachable, gateway down",
        "2026-01-01 10:00:02 ERROR c - could not reach payment service via gateway",
        "2026-01-01 10:00:03 ERROR d - disk write failed, no space left on device",
        "2026-01-01 10:00:04 ERROR e - user record not found in directory",
        "2026-01-01 10:00:05 ERROR f - certificate verify failed for host",
    ])
    clusters = LogAnalytics(LogParser().parse_text(text)).cluster_groups()
    assert clusters, "the three payment-service wordings should form a cluster"
    biggest = max(clusters, key=len)
    assert len(biggest) >= 2
    assert all("payment" in g.signature.lower() for g in biggest)


# -- knowledge base / RAG --------------------------------------------------


def test_knowledge_base_loads():
    entries = load_knowledge_base()
    assert len(entries) >= 10
    assert all(e.entry_id and e.title and e.checks for e in entries)


def test_retrieval_finds_relevant_entry():
    kb = KnowledgeBase(load_knowledge_base())
    hits = kb.retrieve("database connection pool timed out", k=3)
    assert hits
    assert any("database" in entry.subsystem for entry, _ in hits)


def test_retrieval_handles_empty_kb():
    assert KnowledgeBase([]).retrieve("anything") == []


# -- chatbot ---------------------------------------------------------------


@pytest.fixture(scope="module")
def bot() -> LogSense:
    instance = LogSense()
    instance.load_file(SAMPLE_LOG)
    return instance


def test_requires_a_log_before_asking():
    with pytest.raises(RuntimeError):
        LogSense().ask("why did it crash?")


@pytest.mark.parametrize("question", [
    "summarize this log",
    "what are the most frequent errors",
    "what caused the crash",
    "what happened before the first failure",
    "any unusual spikes",
    "are there recurring exceptions",
    "cluster similar errors",
    "show me all database errors",
    "what about the payments service",
])
def test_every_intent_returns_text(bot: LogSense, question: str):
    answer = bot.ask(question, use_llm=False)
    assert isinstance(answer, str) and answer.strip()

def test_root_cause_report_includes_standalone_traceback_details():
    bot = LogSense()
    bot.load_text(
        "Traceback (most recent call last):\n"
        '  File "/repo/vcr/filters.py", line 225, in _multipart_boundary\n'
        "TypeError: startswith first arg must be bytes or a tuple of bytes, not str\n"
    )

    report = bot.root_cause_report()

    assert "TypeError: startswith" in report
    assert 'vcr/filters.py", line 225' in report

def test_root_cause_separates_facts_from_hypotheses(bot: LogSense):
    report = bot.root_cause_report()
    assert "**Observed (from the log)**" in report
    assert "**Possible explanations (not confirmed)**" in report


def test_answers_cite_line_numbers(bot: LogSense):
    assert "L" in bot.ask("most frequent errors", use_llm=False)


def test_comparison_reports_differences(bot: LogSense, tmp_path: Path):
    other = tmp_path / "other.log"
    other.write_text("2026-01-01 10:00:00 ERROR svc - something entirely different\n")
    report = bot.compare_with(other)
    assert "Comparison" in report
    assert "different" in report.lower()
