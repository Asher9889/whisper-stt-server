import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.speech import router as speech_router
from .config import settings
from .services.whisper_service import whisper_service

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("stt")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await whisper_service.load_model()
    logger.info(
        "STT server ready: model=%s device=%s compute_type=%s",
        settings.model_name,
        settings.resolved_device,
        whisper_service.compute_type,
    )
    yield
    await whisper_service.shutdown()
    logger.info("STT server shut down")


app = FastAPI(
    title="STT Whisper Server",
    description=(
        "Self-hosted faster-whisper STT endpoint for the LiveKit voice agent. "
        "Accepts raw mono s16le PCM per utterance via POST /transcribe-pcm."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(speech_router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "stt-whisper",
        "version": "1.0.0",
        "endpoints": ["/transcribe-pcm", "/transcribe", "/v1/stt/health"],
    }
