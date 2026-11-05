# 🔍 LogSense — Application Log Investigation Chatbot

**Ask your application logs what went wrong, and get an answer with line numbers attached.**

When a service falls over, the evidence is already in the log — buried in thousands of
lines of noise, repeated stack traces and timestamps nobody wants to read by hand.
LogSense parses an uploaded log, groups and counts what actually failed, finds the
burst where things went wrong, and answers questions about it in plain language.

Every observed fact it reports is cited by line number. Every explanation is labelled
as a hypothesis, not a finding.

```
you> why did my application crash?

**Observed (from the log)**
- First failure at L11 (14:01:58): db.session - Database connection timeout after 3000ms
- Before it, L9 [WARNING]: db.pool - Connection pool at 9/10 in use, checkout wait 740ms
- Before it, L10 [WARNING]: db.pool - Slow query 2841ms: SELECT * FROM order_items ...
- Highest-volume failure (8x): db.session - Database connection timeout — L11, L20, L21 (+5 more)

**Possible explanations (not confirmed)**
- The first failure carries a stack trace, so the exception origin is visible in the log.
- Failures reference the database subsystem; a database fault is consistent with this pattern.

**Suggested checks**
- Confirm the database accepts connections from the app host on the configured port  [kb-001]
- Compare active connections against the server's max_connections limit  [kb-001]
```

---

## Why this isn't "just send the log to an LLM"

The numbers an LLM quotes back at you are the ones you can least afford to be
invented. So the counts, timings, groupings and correlations here are **computed in
Python**, and the LLM only ever gets to explain evidence it was handed:

| Job | How it's done |
|---|---|
| Find errors, levels, stack traces | `re` — multi-format timestamp + severity parsing |
| Group repeated errors | Message templating (IDs, numbers, paths, quotes → placeholders) |
| Rank what failed most | Frequency analysis over signatures |
| Group *reworded* variants of one error | TF-IDF char n-grams + DBSCAN (cosine) |
| Spot an unusual burst | Per-minute bucketing + z-score |
| Link failures to each other | Time-window correlation + shared-subsystem detection |
| Suggest what to check | TF-IDF retrieval over a troubleshooting knowledge base (RAG) |
| Explain it in prose | LLM, constrained to the evidence above |

**The LLM is optional.** With no API key, LogSense still parses, groups, ranks,
correlates and retrieves — you lose the prose narration, nothing else.

---

## Features

- **Multi-format parsing** — ISO-8601, syslog and Apache-style timestamps, bracketed
  or bare severity tokens, and JSON-lines logs
- **Stack-trace folding** — Python tracebacks and Java `at …` frames stay attached to
  the error that raised them instead of becoming 20 orphan lines
- **Error grouping** — `Timeout for db-1 after 3000ms` and `Timeout for db-7 after
  9000ms` are counted as one problem
- **Error clustering** — differently-worded reports of the same failure get grouped
- **Spike detection** — flags time buckets where the failure rate is statistically unusual
- **Failure correlation** — reports what preceded the first failure, which signature
  appeared earliest, and whether the earliest and loudest failures share a subsystem
- **Subsystem tagging** — database, network, HTTP/API, timeout, auth, memory, disk,
  worker/queue, config
- **Log comparison** — diff the error signatures of two log files
- **Grounded chat** — ask questions, get answers with `L<line>` citations

---

## Prerequisites

- **Python 3.10 or newer** (uses `X | Y` type syntax)
- `pip`
- *Optional:* an `OPENAI_API_KEY`, or any OpenAI-compatible endpoint (Ollama, vLLM,
  OpenRouter) for the prose explanation layer

Developed and tested against Python 3.13, pandas 3.0, scikit-learn 1.9.

---

## Installation

```bash
git clone https://github.com/yathish-debug/logsense-chatbot.git
cd logsense-chatbot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Optional: enable LLM explanations

```bash
export OPENAI_API_KEY="sk-..."           # Windows: set OPENAI_API_KEY=sk-...
export LOGSENSE_MODEL="gpt-4o-mini"      # optional, this is the default
```

Or point it at a local model — no key, no cost:

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export LOGSENSE_MODEL="qwen2.5:7b-instruct"
```

---

## Running it

### Web UI (recommended)

```bash
streamlit run app.py
```

Opens on `http://localhost:8501`. Click **Load sample log** in the sidebar to try it
without uploading anything.

### CLI

```bash
# Interactive session against the bundled sample
python chatbot.py --log data/sample_app.log

# One-shot question
python chatbot.py --log data/sample_app.log --ask "which error should I investigate first?"

# Computed analysis only, no LLM call
python chatbot.py --log data/sample_app.log --ask "summarize this log" --no-llm

# Diff two logs
python chatbot.py --log data/today.log --compare data/yesterday.log
```

In the interactive session: `/summary` `/errors` `/timeline` `/spikes` `/clusters` `/quit`

### Use it as a library

```python
from chatbot import LogSense

bot = LogSense()
bot.load_file("data/sample_app.log")

print(bot.summary())
print(bot.root_cause_report())
print(bot.ask("show me all database errors", use_llm=False))
```

---

## Questions it handles

| Ask | You get |
|---|---|
| *Why did the application crash?* | Observed facts → hypotheses → checks |
| *What are the most frequent errors?* | Ranked signatures with counts and line numbers |
| *What happened before the first failure?* | The preceding warnings and errors in order |
| *Show me all database errors* | Subsystem-filtered entries |
| *Are there recurring exceptions?* | Exception-type tallies and repeated signatures |
| *Any unusual spikes?* | Time buckets with z-scores |
| *Group similar errors* | DBSCAN clusters of related signatures |
| *What about the payments service?* | Keyword search + related troubleshooting guidance |

---

## Project layout

```
logsense-chatbot/
├── chatbot.py                   # Core: parser, analytics, RAG, LLM, CLI
├── app.py                       # Streamlit UI
├── data/
│   ├── troubleshooting_kb.json  # RAG knowledge base (15 entries)
│   └── sample_app.log           # Sample log: a cascading DB failure
├── tests/
│   └── test_chatbot.py          # 30 tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### `chatbot.py` at a glance

| Component | Responsibility |
|---|---|
| `LogRecord` | One logical entry: line number, level, timestamp, continuations |
| `LogParser` | Text/JSON parsing, trace folding, message → signature templating |
| `LogAnalytics` | Counts, grouping, clustering, spike detection, correlation |
| `KnowledgeBase` / `load_knowledge_base` | Loads and retrieves troubleshooting entries |
| `LLMClient` | Optional OpenAI-compatible explanation step |
| `LogSense` | The chatbot: intent routing and grounded answers |

---

## The RAG knowledge base

`data/troubleshooting_kb.json` holds 15 curated entries, each mapping an observable
symptom to likely causes and concrete checks:

```json
{
  "id": "kb-002",
  "title": "Connection pool exhausted",
  "subsystem": "database",
  "symptoms": ["TimeoutError: QueuePool limit of size reached"],
  "likely_causes": ["Connections leaked by code paths that never close sessions"],
  "checks": ["Verify every request path closes or returns its session"],
  "keywords": ["pool", "exhausted", "queuepool", "leak"]
}
```

Retrieval is TF-IDF cosine similarity (`KnowledgeBase.retrieve`), loaded by
`load_knowledge_base()`. A vector database is overkill for a few dozen curated entries,
and TF-IDF keeps the dependency list short and the matches inspectable — every
suggested check is printed with the `kb-` id it came from, so you can audit it.

**Adding your own:** append entries to the JSON — no code change needed. Point at a
different file with `--kb path/to/your_kb.json`.

---

## Running the tests

```bash
pytest -q          # 30 passed
```

Coverage includes stack-trace folding, JSON-lines parsing, severity aliasing,
signature templating, false-positive-resistant subsystem tagging, spike detection
(both firing and staying quiet on flat logs), clustering of reworded variants,
retrieval relevance, and every chat intent.

---

## Design rule

> LogSense distinguishes observed facts from possible explanations. It cites the
> relevant log lines and never claims a suspected root cause is confirmed.

This is enforced in two places: the `root_cause_report()` output is structurally split
into **Observed** / **Possible explanations (not confirmed)** sections, and the LLM
system prompt forbids inventing log lines, requires `L<number>` citations, and requires
hedged causal language.

---

## Limitations

- Analyses **uploaded files**; it does not tail a running application
- Timestamps are parsed as naive local time — mixed-timezone logs are not reconciled
- Correlation is based on ordering and subsystem overlap, so it proposes candidates
  rather than proving causation
- Logs with no timestamps and no severity tokens parse as a single block

---

## License

MIT — see [LICENSE](LICENSE).
