from typing import Any, Generator, TypedDict

import bs4
from backend_base import (
    AppEvent,
    ContextAppEvent,
    HallucinationDetectionEndAppEvent,
    HallucinationDetectionStartAppEvent,
    TextChunkAppEvent,
)
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from lettucedetect_api.client import LettuceClientAsync

_default_system_message = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Don't use emojis."
)


# Define state for application
class State(TypedDict):
    context: list[str]
    question: str
    answer: str
    hallucination_scores: list


def _get_embedding_model(llm_backend: str, config: dict):
    if llm_backend == "OpenAI":
        return OpenAIEmbeddings(model=config["embedding_model"])
    elif llm_backend == "Ollama":
        return OllamaEmbeddings(model=config["embedding_model"], base_url=config["base_url"])
    raise RuntimeError("Unsupported LLM Backend")


def _get_response_model(llm_backend: str, config: dict):
    if llm_backend == "OpenAI":
        return init_chat_model(
            config["response_model"], api_key=config["api_key"], model_provider="openai"
        )
    elif llm_backend == "Ollama":
        return init_chat_model(
            config["response_model"], base_url=config["base_url"], model_provider="ollama"
        )
    raise RuntimeError("unsupported LLM backend")


def make_retrieve(index: VectorStore):
    def retrieve(state: State):
        retrieved_docs = index.similarity_search(state["question"])
        context_str_list = [doc.page_content for doc in retrieved_docs]
        return {"context": context_str_list}

    return retrieve


def make_generate(llm_backend: str, llm_config: dict):
    async def generate(state: State, config: Any):
        llm = _get_response_model(llm_backend, llm_config)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", _default_system_message),
                ("user", "Context: {context}\n\nQuestion: {question}\n"),
            ],
        )
        context_str = "\n\n".join(state["context"])
        messages = prompt_template.invoke({"question": state["question"], "context": context_str})
        response = await llm.ainvoke(messages, config)
        return {
            "answer": response.content,
        }

    return generate


async def detect_hallucination(state: State):
    lettuce_client = LettuceClientAsync("http://127.0.0.1:8000")
    hallucination_scores = await lettuce_client.detect_token(
        contexts=state["context"],
        question=state["question"],
        answer=state["answer"],
    )
    return {
        "hallucination_scores": hallucination_scores.predictions,
    }


async def create_index(url: str, llm_backend: str, config: dict) -> VectorStore:
    embeddings = _get_embedding_model(llm_backend, config)
    vector_store = InMemoryVectorStore(embeddings)
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)
    return vector_store


async def run_query(
    question: str, index: VectorStore, llm_backend: str, config: dict
) -> Generator[AppEvent, None, None]:
    graph_builder = StateGraph(State).add_sequence(
        [make_retrieve(index), make_generate(llm_backend, config), detect_hallucination]
    )
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    async for stream_mode, chunk in graph.astream(
        {"question": question},
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "messages":
            yield TextChunkAppEvent(chunk=chunk[0].content)
        elif stream_mode == "updates":
            if "retrieve" in chunk:
                chunk_data = chunk["retrieve"]
                yield ContextAppEvent(context=chunk_data["context"])
            if "generate" in chunk:
                yield HallucinationDetectionStartAppEvent()
            if "detect_hallucination" in chunk:
                chunk_data = chunk["detect_hallucination"]
                yield HallucinationDetectionEndAppEvent(
                    hallucination_scores=chunk_data["hallucination_scores"],
                )
