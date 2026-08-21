import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Whisper model
    model_name: str = os.getenv("WHISPER_MODEL", "turbo")
    model_dir: str | None = os.getenv("WHISPER_MODEL_DIR") or None
    device: str = os.getenv("WHISPER_DEVICE", "auto")  # auto | cuda | cpu
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

    # Inference knobs (mirrors the reference whisper_service.py)
    beam_size: int = _env_int("WHISPER_BEAM_SIZE", 5)
    temperature: float = _env_float("WHISPER_TEMPERATURE", 0.0)
    vad_filter: bool = _env_bool("WHISPER_VAD_FILTER", False)
    condition_on_previous_text: bool = _env_bool(
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
    )
    language: str | None = os.getenv("WHISPER_LANGUAGE", "hi") or None

    # Audio / resampling
    target_sample_rate: int = _env_int("WHISPER_TARGET_SAMPLE_RATE", 16000)
    min_utterance_ms: int = _env_int("WHISPER_MIN_UTTERANCE_MS", 40)

    # Concurrency / queue
    queue_maxsize: int = _env_int("WHISPER_QUEUE_MAXSIZE", 32)

    # URL transcription API (POST /v1/transcriptions)
    api_token: str | None = os.getenv("STT_API_TOKEN") or None
    download_timeout_s: float = _env_float("STT_DOWNLOAD_TIMEOUT_S", 120.0)
    max_download_mb: int = _env_int("STT_MAX_DOWNLOAD_MB", 100)

    # Debugging
    audio_dump_dir: str | None = os.getenv("WHISPER_AUDIO_DUMP_DIR") or None

    # Server
    host: str = os.getenv("STT_HOST", "0.0.0.0")
    port: int = _env_int("STT_PORT", 8000)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import ctranslate2  # noqa: F401

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"


settings = Settings()
