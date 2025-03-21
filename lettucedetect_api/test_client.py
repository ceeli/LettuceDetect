import time
from multiprocessing import Process
from typing import Generator

import pytest
import uvicorn

from .client import LettuceClient, LettuceClientAsync
from .models import SpanDetectionResponse, TokenDetectionResponse
from .server import app

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


@pytest.fixture(scope="module")
def lettuce_server() -> Generator:
    """Start lettuce detect server for test functions."""
    process = Process(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": SERVER_HOST, "port": SERVER_PORT},
        daemon=True,
    )
    process.start()
    # wait for server startup to complete
    time.sleep(5)
    yield
    process.kill()


def test_valid_token_level_request(lettuce_server: None) -> None:
    """Test valid request for token level detection."""
    client = LettuceClient(SERVER_URL)
    response = client.detect_token(
        contexts=["France is a country in Europe. The capital of France is Paris."],
        question="What is the capital of France? What is the population of France?",
        answer="The capital of France is Paris. The population of France is 69 million.",
    )
    assert isinstance(response, TokenDetectionResponse)
    assert len(response.predictions) >= 1


def test_valid_span_level_request(lettuce_server: None) -> None:
    """Test valid request for token level detection."""
    client = LettuceClient(SERVER_URL)
    response = client.detect_spans(
        contexts=["France is a country in Europe. The capital of France is Paris."],
        question="What is the capital of France? What is the population of France?",
        answer="The capital of France is Paris. The population of France is 69 million.",
    )
    assert isinstance(response, SpanDetectionResponse)
    assert len(response.predictions) >= 1


@pytest.mark.asyncio
async def test_valid_token_level_request_async(lettuce_server: None) -> None:
    """Test valid request for token level detection."""
    client = LettuceClientAsync(SERVER_URL)
    response = await client.detect_token(
        contexts=["France is a country in Europe. The capital of France is Paris."],
        question="What is the capital of France? What is the population of France?",
        answer="The capital of France is Paris. The population of France is 69 million.",
    )
    assert isinstance(response, TokenDetectionResponse)
    assert len(response.predictions) >= 1


@pytest.mark.asyncio
async def test_valid_span_level_request_async(lettuce_server: None) -> None:
    """Test valid request for token level detection."""
    client = LettuceClientAsync(SERVER_URL)
    response = await client.detect_spans(
        contexts=["France is a country in Europe. The capital of France is Paris."],
        question="What is the capital of France? What is the population of France?",
        answer="The capital of France is Paris. The population of France is 69 million.",
    )
    assert isinstance(response, SpanDetectionResponse)
    assert len(response.predictions) >= 1
