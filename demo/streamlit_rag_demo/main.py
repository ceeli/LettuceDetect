import asyncio
import os

import streamlit as st
from backend_base import (
    HallucinationDetectionEndAppEvent,
    HallucinationDetectionStartAppEvent,
    TextChunkAppEvent,
)
from backend_llama_index import create_index, run_query
from dotenv import load_dotenv


def sidebar() -> None:
    logo_url = (
        "https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true"
    )
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-top: -50px; z-index: 1;">
            <img src="{logo_url}" style="width: 200px;"/>
            <h2 style="margin-top: -20px; margin-bottom: 30px;">LettuceDetect</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rag_backend = st.sidebar.selectbox("RAG Backend", ["LangChain", "LlamaIndex"])
    llm_backend = st.sidebar.selectbox("LLM Backend", ["OpenAI", "Ollama"])

    with st.sidebar.expander("LLM Backend Settings", expanded=True):
        if llm_backend == "OpenAI":
            st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""))
            st.text_input("Response Model", value="gpt-4.1-nano")
            st.text_input("Embedding Model", value="text-embedding-3-small")
        elif llm_backend == "Ollama":
            st.text_input("Server Address", value="localhost:11434")
            st.text_input("Response Model", value="llama3.1")
            st.text_input("Embedding Model", value="mxbai-embed-large")

    with st.sidebar.expander("Other Settings"):
        st.text_area("System Message")


async def main_page() -> None:
    if "index_cache" not in st.session_state:
        st.session_state["index_cache"] = {}
    st.subheader("RAG with Hallucination Detection")
    url = st.text_input(
        "URL of Website",
        value="https://lilianweng.github.io/posts/2023-06-23-agent/",
    )
    prompt = st.chat_input("Say something (e.g. What is Task Decomposition?)")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.status("Creating Index...", expanded=True) as status:
                if url not in st.session_state["index_cache"]:
                    print("create index")
                    st.session_state["index_cache"][url] = await create_index(url, {})
                index = st.session_state["index_cache"][url]
                status.update(label="Generating Answer...")
                answer_placeholder = st.empty()
                full_text = ""
                async for app_event in run_query(prompt, index, {}):
                    if isinstance(app_event, TextChunkAppEvent):
                        full_text += app_event.chunk
                        answer_placeholder.markdown(full_text)
                    elif isinstance(app_event, HallucinationDetectionStartAppEvent):
                        status.update(label="Detecting Hallucinations...")
                    elif isinstance(app_event, HallucinationDetectionEndAppEvent):
                        pass
                status.update(label="Answer", state="complete")


async def main():
    load_dotenv()
    st.set_page_config(page_title="RAG with Lettuce")
    sidebar()
    await main_page()


if __name__ == "__main__":
    asyncio.run(main())
