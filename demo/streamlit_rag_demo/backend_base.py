from dataclasses import dataclass

from lettucedetect_api.models import TokenDetectionItem


@dataclass
class AppEvent:
    pass


@dataclass
class TextChunkAppEvent(AppEvent):
    chunk: str


@dataclass
class HallucinationDetectionStartAppEvent(AppEvent):
    pass


@dataclass
class HallucinationDetectionEndAppEvent(AppEvent):
    hallucination_scores: list[TokenDetectionItem]
