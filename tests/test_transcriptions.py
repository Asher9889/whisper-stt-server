import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.whisper_service import whisper_service

URL = "https://storage.example.com/audio/sample.wav"


@pytest.fixture(autouse=True)
def _fake_pipeline(monkeypatch):
    async def fake_load(self):
        pass

    async def fake_shutdown(self):
        pass

    monkeypatch.setattr(whisper_service, "load_model", fake_load)
    monkeypatch.setattr(whisper_service, "shutdown", fake_shutdown)
    monkeypatch.setattr(whisper_service, "_model", object())

    def fake_infer(self, job):
        segments = [
            {"start": 0.0, "end": 4.32, "text": "Welcome to today's class."},
            {"start": 4.32, "end": 9.17, "text": "Today we are going to discuss."},
        ]
        if job.word_timestamps:
            segments = [
                {
                    **seg,
                    "words": [
                        {"text": "word", "start": seg["start"], "end": seg["end"]}
                    ],
                }
                for seg in segments
            ]
        return {
            "transcript": " ".join(s["text"] for s in segments),
            # "de" proves the config default ("hi") is NOT forced on this endpoint
            "language": job.language or "de",
            "confidence": 0.9,
            "durationMs": 268301,
            "segments": segments,
        }

    def fake_decode(self, content):
        return np.zeros(16000, dtype=np.float32)

    async def fake_fetch(url):
        return b"fake-audio-bytes"

    monkeypatch.setattr(whisper_service, "_infer", fake_infer.__get__(whisper_service))
    monkeypatch.setattr(
        whisper_service, "_decode_file", fake_decode.__get__(whisper_service)
    )
    import app.api.transcriptions as mod

    monkeypatch.setattr(mod, "fetch_audio", fake_fetch)


client = TestClient(app)


def _post(json=None, token=None, **overrides):
    payload = {
        "audioUrl": URL,
        "language": "en",
        "wordTimestamps": False,
        **(json or {}),
    }
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/v1/transcriptions", json=payload, headers=headers)


def test_success_segment_timestamps():
    r = _post()
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["durationMs"] == 268301
    assert len(body["segments"]) == 2
    first = body["segments"][0]
    assert set(first) == {"start", "end", "text"}
    assert first["start"] == 0.0
    assert first["end"] == 4.32
    assert first["text"] == "Welcome to today's class."


def test_word_timestamps_included_when_requested():
    r = _post({"wordTimestamps": True})
    assert r.status_code == 200
    seg = r.json()["segments"][0]
    assert seg["words"][0]["start"] == 0.0


def test_language_omitted_uses_detected():
    r = _post({"language": None})
    assert r.status_code == 200
    assert r.json()["language"] == "de"


def test_invalid_scheme_400(monkeypatch):
    from app.services.audio_fetch import fetch_audio as real_fetch
    import app.api.transcriptions as mod

    monkeypatch.setattr(mod, "fetch_audio", real_fetch)
    r = _post({"audioUrl": "ftp://storage.example.com/audio.wav"})
    assert r.status_code == 400


def test_download_failure_400(monkeypatch):
    from app.services.audio_fetch import AudioDownloadError

    async def bad_fetch(url):
        raise AudioDownloadError("Remote server returned HTTP 404 for the audio URL")

    import app.api.transcriptions as mod

    monkeypatch.setattr(mod, "fetch_audio", bad_fetch)
    r = _post()
    assert r.status_code == 400
    assert "404" in r.json()["detail"]


def test_auth_enforced_when_token_configured():
    object.__setattr__(settings, "api_token", "s3cret")
    try:
        assert _post(token="wrong").status_code == 401
        assert _post().status_code == 401
        assert _post(token="s3cret").status_code == 200
    finally:
        object.__setattr__(settings, "api_token", None)


def test_auth_disabled_by_default():
    assert settings.api_token is None
    assert _post().status_code == 200


def test_missing_audio_url_422():
    r = client.post("/v1/transcriptions", json={"language": "en"})
    assert r.status_code == 422


def test_health_still_ok():
    r = client.get("/v1/stt/health")
    assert r.status_code == 200
