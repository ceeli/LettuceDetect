from typing import Type, TypeVar
from urllib.parse import urljoin

import httpx
from models import DetectionRequest, SpanDetectionResponse, TokenDetectionResponse
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _httpx_request_wrapper(
    method: str,
    url: str,
    request: BaseModel,
    response_model: Type[T],
) -> T:
    response = httpx.request(method, url, json=dict(request))
    response.raise_for_status()
    return response_model.model_validate_json(response.text)


async def _httpx_request_wrapper_async(
    method: str,
    url: str,
    request: BaseModel,
    response_model: Type[T],
) -> T:
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, json=dict(request))
    response.raise_for_status()
    return response_model.model_validate_json(response.text)


class LettuceClientBase:
    """Base class class for lettucedetect clients."""

    _TOKEN_ENDPOINT = "/v1/lettucedetect/token"  # noqa: S105
    _SPANS_ENDPOINT = "/v1/lettucedetect/spans"

    def __init__(self, base_url: str):
        """Initialize lettucedetect client sub-classes.

        :param base_url: The full URL of the lettucedetect web server as a
        string. For a local server on port 8000 use "http://127.0.0.1:8000".
        """
        self.base_url = base_url


class LettuceClient(LettuceClientBase):
    """Synchronous client class for lettucedetect web API."""

    def detect_token(
        self, contexts: list[str], question: str, answer: str
    ) -> TokenDetectionResponse:
        """Token-level hallucination detection (synchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `TokenDetectionResponse` pydantic model instance which contains
        the detected tokens in the `predictions` attribute.
        """
        request = DetectionRequest(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._TOKEN_ENDPOINT)
        return _httpx_request_wrapper("post", url, request, TokenDetectionResponse)

    def detect_spans(
        self, contexts: list[str], question: str, answer: str
    ) -> SpanDetectionResponse:
        """Token-level hallucination detection (synchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `SpanDetectionResponse` pydantic model instance which contains
        the detected spans in the `predictions` attribute.
        """
        request = DetectionRequest(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._SPANS_ENDPOINT)
        return _httpx_request_wrapper("post", url, request, SpanDetectionResponse)


class LettuceClientAsync(LettuceClientBase):
    """Asynchronous client class for lettucedetect web API."""

    async def detect_token(
        self, contexts: list[str], question: str, answer: str
    ) -> TokenDetectionResponse:
        """Token-level hallucination detection (asynchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `TokenDetectionResponse` pydantic model instance which contains
        the detected tokens in the `predictions` attribute.
        """
        request = DetectionRequest(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._TOKEN_ENDPOINT)
        return await _httpx_request_wrapper_async("post", url, request, TokenDetectionResponse)

    async def detect_spans(
        self, contexts: list[str], question: str, answer: str
    ) -> SpanDetectionResponse:
        """Token-level hallucination detection (synchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `SpanDetectionResponse` pydantic model instance which contains
        the detected spans in the `predictions` attribute.
        """
        request = DetectionRequest(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._SPANS_ENDPOINT)
        return await _httpx_request_wrapper_async("post", url, request, SpanDetectionResponse)
