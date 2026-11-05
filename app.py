"""Streamlit front end for LogSense.

Run with:  streamlit run app.py

All analysis logic lives in chatbot.py; this module only handles upload,
display and the chat loop.
"""

from __future__ import annotations

import streamlit as st

from chatbot import LogSense

st.set_page_config(page_title="LogSense", page_icon="🔍", layout="wide")

SUGGESTIONS = [
    "What caused the application to crash?",
    "What are the most frequent errors?",
    "What happened before the first failure?",
    "Show me all database errors",
    "Are there any recurring exceptions?",
    "Which error should I investigate first?",
]


@st.cache_resource(show_spinner=False)
def get_bot() -> LogSense:
    """One bot per session; cached so the knowledge base loads once."""
    return LogSense()


def render_timeline(analytics) -> None:
    """Failures per minute, as a bar chart with spikes highlighted."""
    timeline = analytics.failure_timeline()
    if not timeline:
        st.info("No timestamped failures to plot.")
        return
    try:
        import pandas as pd
        import plotly.express as px
    except ImportError:
        st.bar_chart({str(t): c for t, c in timeline})
        return

    spike_times = {start for start, _, _ in analytics.detect_spikes()}
    frame = pd.DataFrame(
        [{"minute": start, "failures": count,
          "kind": "spike" if start in spike_times else "normal"}
         for start, count in timeline])
    figure = px.bar(frame, x="minute", y="failures", color="kind",
                    color_discrete_map={"spike": "#d64545", "normal": "#6b7f99"},
                    title="Failures per minute")
    figure.update_layout(showlegend=False, height=280,
                         margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(figure, use_container_width=True)


def main() -> None:
    bot = get_bot()

    st.title("🔍 LogSense")
    st.caption("Upload an application log and investigate it in plain language. "
               "Observed facts are cited by line number; causes are offered as "
               "hypotheses, never as confirmed findings.")

    with st.sidebar:
        st.header("Log file")
        upload = st.file_uploader("Upload a log", type=["log", "txt", "csv", "json"])
        use_sample = st.button("Load sample log", use_container_width=True)
        st.divider()
        use_llm = st.toggle("Use LLM for explanations", value=bot.llm.available,
                            disabled=not bot.llm.available,
                            help="Set OPENAI_API_KEY (or OPENAI_BASE_URL for a "
                                 "local model) to enable.")
        if not bot.llm.available:
            st.caption("No LLM configured — answers come from computed analysis "
                       "and the troubleshooting knowledge base.")

    # -- load ---------------------------------------------------------------
    if upload is not None:
        text = upload.getvalue().decode("utf-8", errors="replace")
        if st.session_state.get("loaded") != upload.name:
            bot.load_text(text, source=upload.name)
            st.session_state["loaded"] = upload.name
            st.session_state["messages"] = []
    elif use_sample or st.session_state.get("loaded") == "sample_app.log":
        if st.session_state.get("loaded") != "sample_app.log":
            bot.load_file("data/sample_app.log")
            st.session_state["loaded"] = "sample_app.log"
            st.session_state["messages"] = []

    if bot.analytics is None:
        st.info("Upload a log file, or load the sample, to begin.")
        return

    analytics = bot.analytics

    # -- overview -----------------------------------------------------------
    counts = analytics.level_counts
    columns = st.columns(5)
    columns[0].metric("Entries", len(analytics.records))
    columns[1].metric("Errors", counts.get("ERROR", 0))
    columns[2].metric("Critical", counts.get("CRITICAL", 0))
    columns[3].metric("Warnings", counts.get("WARNING", 0))
    columns[4].metric("Stack traces",
                      sum(1 for r in analytics.records if r.has_stack_trace))

    left, right = st.columns([3, 2])
    with left:
        render_timeline(analytics)
    with right:
        st.subheader("Top failures")
        top = analytics.top_groups(5)
        if not top:
            st.success("No ERROR or CRITICAL entries found.")
        for group in top:
            tag = f" · {', '.join(group.subsystems)}" if group.subsystems else ""
            with st.expander(f"({group.count}×) {group.sample[:70]}{tag}"):
                st.code(group.sample, language="text")
                st.caption(f"Lines: {group.cite(10)}")
                if group.exceptions:
                    st.caption("Exceptions: " + ", ".join(group.exceptions))

    with st.expander("AI summary and root-cause analysis"):
        st.markdown(bot.root_cause_report())

    with st.expander("Browse parsed entries"):
        only_problems = st.checkbox("Errors and warnings only", value=True)
        rows = [{"line": r.line_no,
                 "time": r.timestamp.strftime("%H:%M:%S") if r.timestamp else "",
                 "level": r.level,
                 "message": r.message.split("\n")[0][:160]}
                for r in analytics.records
                if not only_problems or r.is_problem]
        st.dataframe(rows, use_container_width=True, height=300)

    # -- chat ---------------------------------------------------------------
    st.divider()
    st.subheader("Ask about this log")

    chips = st.columns(3)
    for index, suggestion in enumerate(SUGGESTIONS):
        if chips[index % 3].button(suggestion, use_container_width=True,
                                   key=f"suggest-{index}"):
            st.session_state["pending"] = suggestion

    st.session_state.setdefault("messages", [])
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("e.g. why did the worker stop?")
    question = question or st.session_state.pop("pending", None)

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Investigating…"):
                answer = bot.ask(question, use_llm=use_llm)
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
