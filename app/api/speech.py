import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile

from ..config import settings
from ..schemas.response import ApiResponse
from ..services.whisper_service import (
    STTBadRequest,
    STTQueueFull,
    whisper_service,
)

logger = logging.getLogger("stt.api")
router = APIRouter(tags=["stt"])

DEFAULT_SAMPLE_RATE = 48000  # LiveKit room rate; used when header is absent.


@router.post("/transcribe-pcm", response_model=ApiResponse)
async def transcribe_pcm(
    request: Request,
    x_sample_rate: Annotated[int | None, Header(alias="X-Sample-Rate")] = None,
    language: Annotated[str | None, Query(description="Language hint, e.g. 'hi'")] = None,
    request_id: Annotated[str | None, Query(description="Correlation ID echoed for logs/tracing")] = None,
    word_timestamps: Annotated[bool, Query(description="Include per-word timestamps")] = False,
) -> ApiResponse:
    """Transcribe a raw mono s16le PCM utterance (no WAV header)."""
    body = await request.body()
    sample_rate = x_sample_rate if x_sample_rate is not None else DEFAULT_SAMPLE_RATE
    loop = asyncio.get_running_loop()
    started = loop.time()

    try:
        data = await whisper_service.transcribe_pcm(
            body,
            sample_rate,
            language=language,
            request_id=request_id,
            word_timestamps=word_timestamps,
        )
    except STTBadRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except STTQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Internal transcription failure request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal transcription failure") from exc

    elapsed_ms = (loop.time() - started) * 1000
    duration_s = len(body) / 2 / sample_rate
    logger.info(
        "transcribe_pcm ok request_id=%s sample_rate=%d duration=%.2fs elapsed=%.0fms transcript=%r",
        request_id,
        sample_rate,
        duration_s,
        elapsed_ms,
        data.get("transcript", ""),
    )
    return ApiResponse(
        success=True,
        message="PCM transcription completed successfully",
        data=data,
    )


@router.post("/transcribe", response_model=ApiResponse)
async def transcribe_file(
    file: Annotated[UploadFile, File(description="Audio file: WAV/MP3/FLAC/OGG/M4A")],
    language: Annotated[str | None, Query(description="Language hint, e.g. 'hi'")] = None,
    request_id: Annotated[str | None, Query(description="Correlation ID echoed for logs/tracing")] = None,
    word_timestamps: Annotated[bool, Query(description="Include per-word timestamps")] = False,
) -> ApiResponse:
    """Offline / batch transcription from an uploaded audio file."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        data = await whisper_service.transcribe_file(
            content,
            language=language,
            request_id=request_id,
            word_timestamps=word_timestamps,
        )
    except STTBadRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except STTQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("File transcription failure request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal transcription failure") from exc

    logger.info(
        "transcribe_file ok request_id=%s file=%s transcript=%r",
        request_id,
        file.filename,
        data.get("transcript", ""),
    )
    return ApiResponse(success=True, message="Transcription completed successfully", data=data)


@router.get("/v1/stt/health")
async def health() -> dict:
    """Liveness probe: model loaded, device, queue depth, request counters."""
    return {
        "status": "ok",
        "model": settings.model_name,
        "model_loaded": whisper_service.model_loaded,
        "device": settings.resolved_device,
        "compute_type": whisper_service.compute_type,
        "language": settings.language,
        "target_sample_rate": settings.target_sample_rate,
        "queue": {
            "depth": whisper_service.queue_depth,
            "maxsize": whisper_service.queue_maxsize,
        },
        "requests": {
            "total": whisper_service.total_requests,
            "failed": whisper_service.failed_requests,
        },
    }
