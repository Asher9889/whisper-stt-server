import asyncio
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.whisper_service import whisper_service


def _pcm(seconds=1.0, rate=48000, freq=440.0):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 20000).astype(np.int16)
    return samples.tobytes()


@pytest.fixture(autouse=True)
def _fake_infer(monkeypatch):
    async def fake_load(self):
        pass

    async def fake_shutdown(self):
        pass

    monkeypatch.setattr(whisper_service, "load_model", fake_load)
    monkeypatch.setattr(whisper_service, "shutdown", fake_shutdown)
    monkeypatch.setattr(whisper_service, "_model", object())

    def fake_infer(self, job):
        return {
            "transcript": "गेहूं में पीला रोग लग गया है",
            "language": job.language or "hi",
            "confidence": 0.92,
        }

    def fake_decode(self, content):
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr(whisper_service, "_infer", fake_infer.__get__(whisper_service))
    monkeypatch.setattr(whisper_service, "_decode_file", fake_decode.__get__(whisper_service))


client = TestClient(app)


def test_success_envelope():
    r = client.post(
        "/transcribe-pcm",
        content=_pcm(),
        headers={"X-Sample-Rate": "48000"},
        params={"request_id": "req-1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["message"] == "PCM transcription completed successfully"
    data = body["data"]
    assert data["transcript"] == "गेहूं में पीला रोग लग गया है"
    assert data["language"] == "hi"
    assert data["confidence"] == 0.92


def test_default_sample_rate_header_absent():
    r = client.post("/transcribe-pcm", content=_pcm())
    assert r.status_code == 200
    assert r.json()["data"]["transcript"]


def test_language_hint_passed():
    r = client.post(
        "/transcribe-pcm", content=_pcm(), headers={"X-Sample-Rate": "48000"},
        params={"language": "en"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["language"] == "en"


def test_empty_body_400():
    r = client.post("/transcribe-pcm", content=b"", headers={"X-Sample-Rate": "48000"})
    assert r.status_code == 400


def test_odd_length_body_400():
    r = client.post("/transcribe-pcm", content=b"\x00", headers={"X-Sample-Rate": "48000"})
    assert r.status_code == 400


def test_bad_sample_rate_400():
    r = client.post("/transcribe-pcm", content=_pcm(), headers={"X-Sample-Rate": "999999"})
    assert r.status_code == 400


def test_too_short_400():
    r = client.post(
        "/transcribe-pcm", content=b"\x00\x00", headers={"X-Sample-Rate": "48000"}
    )
    assert r.status_code == 400


def test_file_endpoint():
    r = client.post(
        "/transcribe",
        files={"file": ("test.wav", _pcm(), "audio/wav")},
        params={"word_timestamps": True},
    )
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_health():
    r = client.get("/v1/stt/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_hallucinated_segments_filtered(monkeypatch):
    from app.services.whisper_service import Job, WhisperService

    class _Seg:
        def __init__(
            self,
            text,
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            compression_ratio=1.0,
        ):
            self.text = text
            self.avg_logprob = avg_logprob
            self.no_speech_prob = no_speech_prob
            self.compression_ratio = compression_ratio
            self.start = 0.0
            self.end = 1.0
            self.words = []

    class _Info:
        language = "hi"
        duration = 3.0

    class _Model:
        def __init__(self, segments):
            self._segments = segments

        def transcribe(self, audio, **kwargs):
            return iter(self._segments), _Info()

    svc = WhisperService()
    svc._model = _Model(
        [
            _Seg("गेहूं में पीला रोग"),
            _Seg("", avg_logprob=-0.1),
            _Seg("thank you", no_speech_prob=0.9, avg_logprob=-1.5),
            _Seg("लगा लगा लगा", compression_ratio=3.0),
        ]
    )

    tone = (
        np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 440) * 0.1
    ).astype(np.float32)
    job = Job(tone, "hi", None, False, None)
    result = svc._infer(job)

    assert result["transcript"] == "गेहूं में पीला रोग"
    assert result["confidence"] > 0


def test_silent_audio_skips_inference():
    from app.services.whisper_service import Job, WhisperService

    class _BrokenModel:
        def transcribe(self, audio, **kwargs):
            raise AssertionError("inference should be skipped for silence")

    svc = WhisperService()
    svc._model = _BrokenModel()
    job = Job(np.zeros(16000, dtype=np.float32), "hi", None, False, None)
    result = svc._infer(job)
    assert result["transcript"] == ""
    assert result["confidence"] == 0.0


def test_invalid_language_400():
    r = client.post(
        "/transcribe-pcm",
        content=_pcm(),
        headers={"X-Sample-Rate": "48000"},
        params={"language": "xx-YY"},
    )
    assert r.status_code == 400
    assert "Invalid language code" in r.json()["detail"]
    assert "xx-YY" in r.json()["detail"]


def test_invalid_config_default_language_raises_400_before_inference():
    from app.config import Settings
    from app.services.whisper_service import (
        Job,
        STTBadRequest,
        WhisperService,
    )

    svc = WhisperService(cfg=Settings(language="xx-YY"))

    class _BrokenModel:
        def transcribe(self, audio, **kwargs):
            raise AssertionError("should be rejected before inference")

    svc._model = _BrokenModel()
    tone = (
        np.sin(np.linspace(0, 1, 16000) * 2 * np.pi * 440) * 0.1
    ).astype(np.float32)
    job = Job(tone, None, None, False, None)
    with pytest.raises(STTBadRequest, match="xx-YY"):
        svc._infer(job)


def test_concurrent_requests_serialized():
    import threading

    from app.services.whisper_service import whisper_service as svc
    from httpx import ASGITransport, AsyncClient

    gate = threading.Event()

    def slow_infer(self, job):
        gate.wait(5)
        return {"transcript": "x", "language": "hi", "confidence": 1.0}

    async def run():
        orig = svc._infer
        svc._infer = slow_infer.__get__(svc)
        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                tasks = [
                    asyncio.create_task(
                        ac.post(
                            "/transcribe-pcm",
                            content=_pcm(),
                            headers={"X-Sample-Rate": "48000"},
                        )
                    )
                    for _ in range(5)
                ]
                await asyncio.sleep(0.05)
                assert svc.queue_depth > 0, "requests should be queued, not concurrent"
                gate.set()
                results = await asyncio.gather(*tasks)
                assert all(r.status_code == 200 for r in results)
        finally:
            svc._infer = orig

    asyncio.run(run())
