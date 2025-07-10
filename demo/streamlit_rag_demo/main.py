import asyncio
import os

import streamlit as st
from backend_base import (
    ContextAppEvent,
    HallucinationDetectionEndAppEvent,
    HallucinationDetectionStartAppEvent,
    TextChunkAppEvent,
)
from backend_llama_index import create_index, run_query
from dotenv import load_dotenv

from lettucedetect_api.models import TokenDetectionItem


def _fix_whitespace(text: str) -> str:
    return '<span style="white-space: pre-wrap;">' + text + "</span>"


def _get_html_output(predictions: list[TokenDetectionItem]) -> str:
    text = [item.token for item in predictions]

    colors = [f"rgba(255, 0, 0, {item.hallucination_score * 0.7})" for item in predictions]
    html_elements = [
        f'<span style="background-color: {color};">{text}</span>'
        for color, text in zip(colors, text)
    ]
    html = "".join(html_elements)
    return html


def _get_lettuce_detect_logo() -> str:
    logo_url = (
        "https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true"
    )
    return f"""
    <div style="text-align: center; margin-top: -50px; z-index: 1;">
        <img src="{logo_url}" style="width: 200px;"/>
        <h2 style="margin-top: -20px; margin-bottom: 30px;">LettuceDetect</h2>
    </div>
    """


def _draw_sidebar() -> dict:
    config = {}
    st.sidebar.markdown(_get_lettuce_detect_logo(), unsafe_allow_html=True)
    config["llm_backend"] = st.sidebar.selectbox("LLM Backend", ["OpenAI", "Ollama"])
    s = {}
    with st.sidebar.expander("LLM Backend Settings", expanded=True):
        if config["llm_backend"] == "OpenAI":
            env_api_key = os.environ.get("OPENAI_API_KEY", "")
            s["api_key"] = st.text_input("OpenAI API Key", value=env_api_key)
            s["response_model"] = st.text_input("Response Model", value="gpt-4.1-nano")
            s["embedding_model"] = st.text_input("Embedding Model", value="text-embedding-3-small")
        elif config["llm_backend"] == "Ollama":
            s["base_url"] = st.text_input("Server Address", value="localhost:11434")
            s["response_model"] = st.text_input("Response Model", value="llama3.2")
            s["embedding_model"] = st.text_input("Embedding Model", value="mxbai-embed-large")

    config["llm_backend_settings"] = s
    return config


def _draw_header() -> str:
    st.subheader("Website RAG")
    return st.text_input(
        "URL of Website",
        value="https://lilianweng.github.io/posts/2023-06-23-agent/",
    )


async def _draw_chat(url: str, question: str, config: dict) -> None:
    llm_backend = config["llm_backend"]
    llm_backend_config = config.get("llm_backend_settings", {})
    with st.chat_message("user"):
        st.write(question)
        st.session_state["last_prompt"] = question

    with st.chat_message("assistant"):
        with st.status("Creating Index...") as status:
            index_cache_key = (llm_backend, url)
            if index_cache_key not in st.session_state["index_cache"]:
                st.session_state["index_cache"][index_cache_key] = await create_index(
                    url, llm_backend, llm_backend_config
                )
            index = st.session_state["index_cache"][index_cache_key]

            status.update(label="Fetching Context...")
            event_iter = run_query(question, index, llm_backend, llm_backend_config)
            async for app_event in event_iter:
                if isinstance(app_event, ContextAppEvent):
                    next_chunk_str = "\n\n-- NEXT CHUNK --\n\n"
                    header = f"**Nr. of chunks: {len(app_event.context)}**{next_chunk_str}"
                    context_text = next_chunk_str.join(app_event.context)
                    st.markdown(header + "\n\n" + context_text)
                    break
            status.update(label="Context", state="complete")
        with st.status("Generating Answer...", expanded=True) as status:
            answer_placeholder = None
            full_text = ""
            async for app_event in event_iter:
                if isinstance(app_event, TextChunkAppEvent):
                    if app_event.chunk != "":
                        full_text += app_event.chunk
                        if answer_placeholder is None:
                            answer_placeholder = st.empty()
                        answer_placeholder.html(_fix_whitespace(full_text))
                elif isinstance(app_event, HallucinationDetectionStartAppEvent):
                    status.update(label="Detecting Hallucinations...")
                elif isinstance(app_event, HallucinationDetectionEndAppEvent):
                    html = _get_html_output(app_event.hallucination_scores)
                    if answer_placeholder is None:
                        answer_placeholder = st.empty()
                    answer_placeholder.html(_fix_whitespace(html))
                    st.session_state["last_answer"] = app_event.hallucination_scores
            status.update(label="Answer", state="complete")


async def _draw_chat_cached() -> None:
    with st.chat_message("user"):
        st.write(st.session_state["last_prompt"])
    with st.chat_message("assistant"):
        with st.status("Context", state="complete"):
            st.text("Hi")
        with st.status("Answer", expanded=True, state="complete"):
            html = _get_html_output(st.session_state["last_answer"])
            answer_placeholder = st.empty()
            answer_placeholder.html(_fix_whitespace(html))


async def _main() -> None:
    # Initialize cache of RAG index as Streamlit session variable.
    if "index_cache" not in st.session_state:
        st.session_state["index_cache"] = {}

    # Load environment file for API keys.
    load_dotenv()
    st.set_page_config(page_title="LettuceDetect RAG Demo")
    config = _draw_sidebar()
    url = _draw_header()

    question = st.chat_input("Ask something (e.g. What is Task Decomposition?)")
    if question:
        await _draw_chat(url, question, config)
    elif "last_prompt" in st.session_state and "last_answer" in st.session_state:
        # Required so that the chat doesn't disappear when inputs are modified
        # in the sidebar or the header.
        await _draw_chat_cached()


if __name__ == "__main__":
    # Use async because RAG Frameworks are built for it.
    # Makes token and event streaming possible/easier.
    asyncio.run(_main())
