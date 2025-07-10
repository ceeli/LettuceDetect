import asyncio
from typing import Generator

from backend_base import (
    AppEvent,
    ContextAppEvent,
    HallucinationDetectionEndAppEvent,
    HallucinationDetectionStartAppEvent,
    TextChunkAppEvent,
)
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader

from lettucedetect_api.client import LettuceClientAsync

_default_system_message = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Don't use emojis."
)

_prompt_template = RichPromptTemplate(
    '{% chat role="system" %}\n' + _default_system_message + "{% endchat %}\n"
    '{% chat role="user" %}\n'
    "Context: {{ context_str }}\n\nQuestion: {{ query_str }}\n"
    "{% endchat %}\n"
)


class _AnswerEvent(Event):
    """Result of a RAG query."""

    question: str
    answer: str
    context: list[str]


class _TextChunkEvent(Event):
    chunk: str


class _ContextEvent(Event):
    context: list[str]


class _HallucinationDetectionStartEvent(Event):
    pass


def _get_embedding_model(llm_backend: str, config: dict):
    if llm_backend == "OpenAI":
        return OpenAIEmbedding(model_name=config["embedding_model"], api_key=config["api_key"])
    elif llm_backend == "Ollama":
        return OllamaEmbedding(model_name=config["embedding_model"], base_url=config["base_url"])
    raise RuntimeError("Unsupported LLM Backend")


def _get_response_model(llm_backend: str, config: dict):
    if llm_backend == "OpenAI":
        return OpenAI(model=config["response_model"], api_key=config["api_key"])
    elif llm_backend == "Ollama":
        return Ollama(model=config["response_model"], base_url=config["base_url"])
    raise RuntimeError("Unsupported LLM Backend")


async def create_index(url: str, llm_backend: str, config: dict) -> VectorStoreIndex:
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=60)
    embed_model = _get_embedding_model(llm_backend, config)
    return VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=embed_model,
        transformations=[splitter],
    )


async def run_query(
    question: str, index: VectorStoreIndex, llm_backend: str, config: dict
) -> Generator[AppEvent, None, None]:
    workflow = RAGWorkflow()
    handler = workflow.run(index=index, question=question, llm_backend=llm_backend, config=config)
    async for ev in handler.stream_events():
        if isinstance(ev, _TextChunkEvent):
            yield TextChunkAppEvent(chunk=ev.chunk)
        elif isinstance(ev, _ContextEvent):
            yield ContextAppEvent(context=ev.context)
        elif isinstance(ev, _HallucinationDetectionStartEvent):
            yield HallucinationDetectionStartAppEvent()
        elif isinstance(ev, StopEvent):
            yield HallucinationDetectionEndAppEvent(
                hallucination_scores=ev.result["hallucination_scores"],
            )


class RAGWorkflow(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> _AnswerEvent:
        question = ev.get("question")
        index = ev.get("index")
        llm_backend = ev.get("llm_backend")
        config = ev.get("config")
        llm = _get_response_model(llm_backend, config)
        retriever = index.as_retriever(similarity_top_k=2)
        synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="simple_summarize",
            text_qa_template=_prompt_template,
            streaming=True,
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )
        # streaming_response = query_engine.query(question)
        source_nodes = retriever.retrieve(question)
        context = [n.node.text for n in source_nodes]
        ctx.write_event_to_stream(_ContextEvent(context=context))
        streaming_response = synthesizer.synthesize(question, nodes=source_nodes)
        full_response = []
        for chunk in streaming_response.response_gen:
            ctx.write_event_to_stream(_TextChunkEvent(chunk=chunk))
            await asyncio.sleep(0)
            full_response.append(chunk)
        return _AnswerEvent(question=question, answer="".join(full_response), context=context)

    @step
    async def detect_hallucination(self, ctx: Context, ev: _AnswerEvent) -> StopEvent:
        ctx.write_event_to_stream(_HallucinationDetectionStartEvent())
        lettuce_client = LettuceClientAsync("http://127.0.0.1:8000")
        hallucination_scores = await lettuce_client.detect_token(
            contexts=ev.context, question=ev.question, answer=ev.answer
        )
        return StopEvent(
            result={
                "answer": ev.answer,
                "hallucination_scores": hallucination_scores.predictions,
            }
        )
