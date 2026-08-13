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
