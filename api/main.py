import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from lettucedetect.models.inference import HallucinationDetector


class Settings(BaseSettings):
    lettucedetect_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    lettucedetect_method: str = "transformer"


class DetectionRequest(BaseModel):
    contexts: list[str]
    question: str
    answer: str


class TokenDetectionItem(BaseModel):
    token: str
    hallucination_score: float


class SpanDetectionItem(BaseModel):
    start: int
    end: int
    text: str
    hallucination_score: float


class TokenDetectionResponse(BaseModel):
    predictions: list[TokenDetectionItem]


class SpanDetectionResponse(BaseModel):
    predictions: list[SpanDetectionItem]


settings = Settings()
detector: HallucinationDetector | None = None


@asynccontextmanager
async def init_detector(app: FastAPI) -> AsyncIterator[None]:
    """Load model and initialize hallucination detector object.

    This fastapi livespan event is run once during fastapi startup. It is used
    to load and initialize the hallucination detector. All subsequent requests
    can then use the hallucination detector without repeating the initialization
    steps over and over again.

    :param app: The FastAPI object for this livespan event.
    """
    global detector
    detector = HallucinationDetector(
        method=settings.lettucedetect_method,
        model_path=settings.lettucedetect_model,
    )
    yield


app = FastAPI(lifespan=init_detector)
detector_lock = asyncio.Lock()


async def run_detector_safe(request: DetectionRequest, output_format: str):
    async with detector_lock:
        preds = await run_in_threadpool(
            detector.predict,
            context=request.contexts,
            question=request.question,
            answer=request.answer,
            output_format=output_format,
        )
    return preds


@app.post("/lettucedetect/token", response_model=TokenDetectionResponse)
async def run_token_detection(request: DetectionRequest) -> dict:
    preds = await run_detector_safe(request, output_format="tokens")
    preds_converted = [{"token": p["token"], "hallucination_score": p["prob"]} for p in preds]
    return {"predictions": preds_converted}


@app.post("/lettucedetect/spans", response_model=SpanDetectionResponse)
async def run_span_detection(request: DetectionRequest) -> dict:
    preds = await run_detector_safe(request, output_format="spans")
    preds_converted = [
        {
            "start": p["start"],
            "end": p["end"],
            "text": p["text"],
            "hallucination_score": p["confidence"],
        }
        for p in preds
    ]
    return {"predictions": preds_converted}
