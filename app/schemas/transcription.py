from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Body of POST /v1/transcriptions."""

    audioUrl: str = Field(
        min_length=8,
        max_length=2048,
        description="HTTP(S) URL of the audio file",
    )
    language: str | None = Field(
        default=None,
        max_length=32,
        description="ISO code or name, e.g. 'en', 'hi'. Omit to auto-detect.",
    )
    wordTimestamps: bool = Field(
        default=False,
        description="Also include per-word timestamps inside each segment.",
    )


class WordOut(BaseModel):
    text: str
    start: float
    end: float


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordOut] | None = None


class TranscriptionResponse(BaseModel):
    language: str
    durationMs: int
    segments: list[SegmentOut]
