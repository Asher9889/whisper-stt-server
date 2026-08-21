import asyncio
import logging
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from ..config import settings
from ..schemas.transcription import SegmentOut, TranscriptionRequest, TranscriptionResponse
from ..services.audio_fetch import AudioDownloadError, fetch_audio
from ..services.whisper_service import (
    STTBadRequest,
    STTQueueFull,
    whisper_service,
)

logger = logging.getLogger("stt.api")
router = APIRouter(tags=["transcriptions"])


async def require_bearer(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Validate `Authorization: Bearer <token>` when STT_API_TOKEN is set."""
    expected = settings.api_token
    if not expected:  # auth disabled (dev)
        return
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        supplied.strip(), expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/v1/transcriptions",
    response_model=TranscriptionResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_bearer)],
)
async def create_transcription(req: TranscriptionRequest) -> TranscriptionResponse:
    """Download audio from `audioUrl` and transcribe it into timestamped segments."""
    url = str(req.audioUrl)
    loop = asyncio.get_running_loop()
    started = loop.time()

    try:
        content = await fetch_audio(url)
        audio = await whisper_service.decode_audio_bytes(content)
        data = await whisper_service.transcribe_audio(
            audio,
            language=req.language,
            word_timestamps=req.wordTimestamps,
        )
    except AudioDownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except STTBadRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except STTQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("URL transcription failure url=%s", url)
        raise HTTPException(status_code=500, detail="Internal transcription failure") from exc

    elapsed_ms = (loop.time() - started) * 1000
    logger.info(
        "transcribe_url ok url=%s language=%s segments=%d duration_ms=%d elapsed=%.0fms",
        url,
        data["language"],
        len(data.get("segments", [])),
        data.get("durationMs", 0),
        elapsed_ms,
    )
    return TranscriptionResponse(
        language=data["language"],
        durationMs=data["durationMs"],
        segments=[SegmentOut(**seg) for seg in data["segments"]],
    )
