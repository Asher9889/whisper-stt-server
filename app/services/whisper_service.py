import asyncio
import io
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from ..config import Settings, settings

logger = logging.getLogger("stt.whisper")

# Domain vocabulary (Hindi agriculture) used as initial_prompt — the same
# quality lever as the reference whisper_service.py.
DEFAULT_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    (
        "गेहूं में पीला रोग, धान में भूरा धब्बा, मक्का, सोयाबीन, बाजरा, ज्वार, चना, "
        "मूंग, उड़द, अरहर, तिल, सरसों, मूंगफली, कपास, गन्ना, आलू, टमाटर, प्याज, "
        "लहसुन, मिर्च, धनिया, गोभी, बैंगन, भिंडी, करेला, कद्दू, खरबूजा, कीटनाशक, "
        "खाद, उर्वरक, सिंचाई, बीज, फसल, किसान, मिट्टी, रोग, कीट, दवा, छिड़काव, "
        "उपज, मंडी, भाव, फसल बीमा, मौसम, वर्षा, सूखा, बारिश"
    ),
)


class STTError(Exception):
    """Base class for STT server errors."""


class STTBadRequest(STTError):
    """Invalid request payload -> HTTP 400."""


class STTQueueFull(STTError):
    """Inference queue is full -> HTTP 429 (retryable)."""


@dataclass
class Job:
    audio: np.ndarray  # float32 mono PCM at target_sample_rate, [-1, 1]
    language: str | None
    request_id: str | None
    word_timestamps: bool
    future: asyncio.Future
    queued_at: float = field(default_factory=time.monotonic)


class WhisperService:
    """Singleton wrapper around a faster-whisper model.

    - Keeps the model resident in memory (load once at startup).
    - Serializes all GPU inference through a single asyncio worker, so
      concurrent / overlapping ``recognize()`` calls are queued instead of
      failing or thrashing the GPU (spec §4.2).
    - Client disconnects cancel the caller's future; the worker skips
      cancelled jobs so no inference is wasted (spec §4.4).
    """

    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self._model = None
        self._load_lock = asyncio.Lock()
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=cfg.queue_maxsize)
        self._worker_task: asyncio.Task | None = None
        self.total_requests = 0
        self.failed_requests = 0

    # ------------------------------------------------------------- model ---
    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_maxsize(self) -> int:
        return self.cfg.queue_maxsize

    @property
    def compute_type(self) -> str:
        return self._resolved_compute_type(self.cfg.resolved_device)

    def _resolved_compute_type(self, device: str) -> str:
        ct = self.cfg.compute_type
        if device == "cpu" and ct == "float16":
            logger.warning("float16 is not supported on CPU; falling back to int8")
            return "int8"
        return ct

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        device = self.cfg.resolved_device
        compute_type = self._resolved_compute_type(device)
        kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
        if self.cfg.model_dir:
            kwargs["download_root"] = self.cfg.model_dir
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.cfg.model_name,
            device,
            compute_type,
        )
        self._model = WhisperModel(self.cfg.model_name, **kwargs)
        logger.info("Model loaded")

    async def load_model(self) -> None:
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_model)
        self._ensure_worker()

    async def shutdown(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    # ------------------------------------------------------------ worker ---
    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="whisper-inference-worker"
            )

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job.future.cancelled():
                    continue
                result = await asyncio.to_thread(self._infer, job)
                if not job.future.cancelled():
                    job.future.set_result(result)
            except asyncio.CancelledError:
                if not job.future.cancelled():
                    job.future.cancel()
                raise
            except Exception as exc:  # worker must survive any failure
                self.failed_requests += 1
                logger.exception("Transcription failed request_id=%s", job.request_id)
                if not job.future.cancelled():
                    job.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _enqueue(self, job: Job) -> dict[str, Any]:
        self._ensure_worker()
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise STTQueueFull("STT inference queue is full; retry later") from exc
        try:
            return await job.future
        except asyncio.CancelledError:
            job.future.cancel()
            raise

    # ------------------------------------------------------------ public ---
    async def transcribe_pcm(
        self,
        pcm: bytes,
        sample_rate: int,
        language: str | None = None,
        request_id: str | None = None,
        word_timestamps: bool = False,
    ) -> dict[str, Any]:
        if not pcm:
            raise STTBadRequest("Empty PCM body")
        if len(pcm) % 2 != 0:
            raise STTBadRequest("PCM body must contain whole int16 samples")
        if not 8000 <= sample_rate <= 96000:
            raise STTBadRequest(f"Invalid sample rate: {sample_rate}")

        duration_s = len(pcm) / 2 / sample_rate
        min_s = self.cfg.min_utterance_ms / 1000.0
        if duration_s < min_s:
            raise STTBadRequest(
                f"PCM too short: {duration_s:.3f}s < {self.cfg.min_utterance_ms}ms"
            )

        samples = np.frombuffer(pcm, dtype=np.int16)
        audio = samples.astype(np.float32) / 32768.0
        if sample_rate != self.cfg.target_sample_rate:
            audio = resample_poly(
                audio, up=self.cfg.target_sample_rate, down=sample_rate
            )
        self._dump_audio(samples, sample_rate, request_id)

        future = asyncio.get_running_loop().create_future()
        return await self._enqueue(
            Job(audio, language, request_id, word_timestamps, future)
        )

    async def transcribe_file(
        self,
        content: bytes,
        language: str | None = None,
        request_id: str | None = None,
        word_timestamps: bool = False,
    ) -> dict[str, Any]:
        audio = await asyncio.to_thread(self._decode_file, content)
        if audio.size == 0:
            raise STTBadRequest("No decodable audio found in the uploaded file")

        future = asyncio.get_running_loop().create_future()
        return await self._enqueue(
            Job(audio, language, request_id, word_timestamps, future)
        )

    # ------------------------------------------------------------- impl ---
    def _decode_file(self, content: bytes) -> np.ndarray:
        from faster_whisper.audio import decode_audio

        return decode_audio(
            io.BytesIO(content), sampling_rate=self.cfg.target_sample_rate
        )

    def _dump_audio(
        self, samples: np.ndarray, sample_rate: int, request_id: str | None
    ) -> None:
        dump_dir = self.cfg.audio_dump_dir
        if not dump_dir:
            return
        os.makedirs(dump_dir, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{request_id or os.getpid()}.wav"
        path = os.path.join(dump_dir, name)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        logger.debug("Dumped %d samples @%dHz to %s", samples.size, sample_rate, path)

    def _infer(self, job: Job) -> dict[str, Any]:
        self.total_requests += 1
        model = self._model
        if model is None:
            raise STTError("Whisper model is not loaded")

        used_language = job.language or self.cfg.language
        segments_iter, info = model.transcribe(
            job.audio,
            language=used_language,
            beam_size=self.cfg.beam_size,
            temperature=self.cfg.temperature,
            vad_filter=self.cfg.vad_filter,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            initial_prompt=DEFAULT_INITIAL_PROMPT,
            word_timestamps=job.word_timestamps,
        )
        segments = list(segments_iter)

        texts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
        transcript = " ".join(texts)

        # Confidence from avg log-prob as in the reference impl: exp(avg_lp).
        confidence = (
            float(np.exp(np.mean([seg.avg_logprob for seg in segments])))
            if segments
            else 0.0
        )
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        result: dict[str, Any] = {
            "transcript": transcript,
            "language": used_language or info.language,
            "confidence": confidence,
        }
        if job.word_timestamps:
            result["words"] = [
                {
                    "text": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                }
                for seg in segments
                for w in (seg.words or [])
            ]
        return result


whisper_service = WhisperService()
