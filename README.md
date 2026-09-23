# STT Whisper Server

Self-hosted **faster-whisper** Speech-to-Text server for the Krishi voice agent.
It exposes one HTTP contract tailored to two very different consumers:

1. **LiveKit voice agent path** — one VAD-segmented utterance per request, raw
   mono `s16le` PCM, wrapped in a `{success, message, data}` envelope
   (`POST /transcribe-pcm`).
2. **Offline / batch path** — download an audio file from a URL and return
   segment-level and optional word-level timestamps (`POST /v1/transcriptions`).

The model is loaded **once** at startup, kept resident in memory, and all GPU
inference is serialized through a single asyncio worker + queue so concurrent
requests are queued rather than thrashing the GPU or failing.

---

## Table of contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Configuration (environment variables)](#configuration-environment-variables)
- [How the service works](#how-the-service-works)
- [Endpoints](#endpoints)
  - [GET /](#get-)
  - [POST /transcribe-pcm](#post-transcribe-pcm-livekit-client-path)
  - [POST /transcribe](#post-transcribe-offline-file-upload)
  - [POST /v1/transcriptions](#post-v1transcriptions-audio-url--timestamped-segments)
  - [GET /v1/stt/health](#get-v1stthealth)
- [Response schemas](#response-schemas)
- [Error handling](#error-handling)
- [Design notes](#design-notes)
- [Testing](#testing)
- [Interactive API docs](#interactive-api-docs)

---

## Requirements

- **Python 3.10** (pinned in `.python-version`).
- Linux or macOS. CUDA is optional; if present it is auto-detected, otherwise
  the service runs on CPU.
- ~4 GB+ RAM depending on the selected model (`large-v3` is recommended).

Dependencies (`requirements.txt`):

| Package | Purpose |
|---|---|
| `fastapi>=0.111.0` | HTTP framework / routing / validation |
| `uvicorn[standard]>=0.30.0` | ASGI server |
| `faster-whisper>=1.0.3` | Whisper inference (CTranslate2 backend) + audio decoding |
| `numpy>=1.26.0` | PCM handling |
| `scipy>=1.12.0` | Polyphase resampling (`resample_poly`) |
| `python-multipart>=0.0.9` | Multipart form parsing for file uploads |
| `httpx>=0.27.0` | Async HTTP client for downloading audio URLs |

> Tests additionally require `pytest` and `httpx`'s ASGI transport (bundled with
> `httpx`).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: adjust settings
python run.py
```

Server listens on `0.0.0.0:8000` by default (change `STT_HOST` / `STT_PORT`).

The model is downloaded from HuggingFace on **first start** and cached
(optionally under `WHISPER_MODEL_DIR`). Subsequent starts reuse the cache.
`run.py` reads `STT_HOST`, `STT_PORT` and `LOG_LEVEL` and passes them to
uvicorn; the app itself is `app.main:app`.

Equivalent direct uvicorn invocation:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Selected models: `tiny`, `base`, `small`, `medium`, `large-v3` (**default**), or
`turbo`. On a CPU-only machine `float16` automatically falls back to `int8`.

---

## Project layout

```
stt-whisper/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Settings dataclass, all env parsing
│   ├── main.py                   # FastAPI app, lifespan, router wiring, GET /
│   ├── api/
│   │   ├── speech.py             # POST /transcribe-pcm, POST /transcribe, GET /v1/stt/health
│   │   └── transcriptions.py     # POST /v1/transcriptions + bearer auth
│   ├── schemas/
│   │   ├── response.py           # ApiResponse envelope {success, message, data}
│   │   └── transcription.py      # TranscriptionRequest / Response / Segment / Word
│   └── services/
│       ├── audio_fetch.py        # URL audio download with scheme + size limits
│       └── whisper_service.py    # WhisperService singleton, queue + worker, inference
├── tests/
│   ├── test_api.py               # /transcribe-pcm, /transcribe, health, concurrency
│   └── test_transcriptions.py    # /v1/transcriptions, auth, URL validation
├── run.py                        # uvicorn entrypoint
├── requirements.txt
├── .env.example                  # documented env template
└── .python-version               # 3.10.20
```

---

## Configuration (environment variables)

All settings live in `app/config.py` (`Settings` dataclass, instance
`settings`). Values are read from the environment (or `.env` if you export it)
at import time. Invalid integers/floats silently fall back to their default.

### Whisper model

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Model name/size: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`. |
| `WHISPER_MODEL_DIR` | *(none)* | Local directory to cache/download the model into. |
| `WHISPER_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu`. `auto` probes CTranslate2 for a CUDA device and falls back to `cpu` on any error. |
| `WHISPER_COMPUTE_TYPE` | `float16` | CTranslate2 compute type. On CPU, `float16` is automatically downgraded to `int8` (a warning is logged). |

### Inference knobs

| Variable | Default | Description |
|---|---|---|
| `WHISPER_BEAM_SIZE` | `5` | Beam search width. |
| `WHISPER_TEMPERATURE` | `0.0` | Sampling temperature (`0.0` = deterministic). |
| `WHISPER_VAD_FILTER` | `false` | Whisper's internal VAD. **Off by default** because the LiveKit agent already VAD-segments utterances. |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `false` | Whether to condition on previously decoded text. |
| `WHISPER_LANGUAGE` | `hi` | Default language hint for `/transcribe-pcm` and `/transcribe`. Empty string is treated as unset (auto-detect). |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | A segment is dropped if `no_speech_prob` exceeds this **and** `avg_logprob` is below `WHISPER_LOGPROB_THRESHOLD`. |
| `WHISPER_LOGPROB_THRESHOLD` | `-1.0` | Low-confidence cutoff used together with `WHISPER_NO_SPEECH_THRESHOLD`. |
| `WHISPER_COMPRESSION_RATIO_THRESHOLD` | `2.4` | Segments with a gzip compression ratio above this (repetitive loops) are dropped. |
| `WHISPER_SILENCE_LEVEL` | `0.0001` | Peak signal amplitude in `[-1, 1]` below this → treated as digital silence, inference is skipped and an empty transcript is returned. Catches clips Whisper would otherwise hallucinate on (e.g. `no_speech_prob ≈ 0` for full digital silence). |
| `WHISPER_INITIAL_PROMPT` | *(built-in Hindi agriculture vocabulary)* | Read directly in `whisper_service.py` (not via `Settings`). Not applied by default — callers can pass `initial_prompt` per request to use domain vocabulary. See [Design notes](#domain-initial-prompt). |

### Audio

| Variable | Default | Description |
|---|---|---|
| `WHISPER_TARGET_SAMPLE_RATE` | `16000` | Rate the model expects; incoming PCM is resampled to this. |
| `WHISPER_MIN_UTTERANCE_MS` | `40` | PCM shorter than this is rejected as `400` (avoid feeding noise/zero-length clips to the model). |
| `WHISPER_AUDIO_DUMP_DIR` | *(none)* | If set, every `/transcribe-pcm` body is resampled to `WHISPER_TARGET_SAMPLE_RATE` (16 kHz), then written as a mono WAV for debugging. |

### Concurrency

| Variable | Default | Description |
|---|---|---|
| `WHISPER_QUEUE_MAXSIZE` | `32` | Max queued inference jobs. When full, new requests get `429`. |

### URL transcription API

| Variable | Default | Description |
|---|---|---|
| `STT_API_TOKEN` | *(none)* | When set, `POST /v1/transcriptions` requires `Authorization: Bearer <token>`. **Empty = auth disabled (dev only).** |
| `STT_DOWNLOAD_TIMEOUT_S` | `120` | Total timeout for the remote audio download. |
| `STT_MAX_DOWNLOAD_MB` | `100` | Max audio download size in MB (enforced from `Content-Length` and while streaming). |

### Server

| Variable | Default | Description |
|---|---|---|
| `STT_HOST` | `0.0.0.0` | Bind address (used by `run.py`). |
| `STT_PORT` | `8000` | Bind port (used by `run.py`). |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG` also logs audio-dump paths). |

**Booleans** accept `1`, `true`, `yes`, `on` (case-insensitive).

---

## How the service works

Startup (`app/main.py::lifespan`):

1. `whisper_service.load_model()` loads the model in a thread (blocking work is
   off-loaded via `asyncio.to_thread`) and starts the single inference worker.
2. A ready log line is emitted with `model`, resolved `device` and
   `compute_type`.
3. On shutdown the worker task is cancelled.

Inference pipeline:

1. An endpoint validates input and produces a `Job` containing `float32` mono
   audio at the target sample rate, plus `language`, `request_id`,
   `word_timestamps`, and an `asyncio.Future`.
2. `WhisperService._enqueue` does a non-blocking `put_nowait` on an
   `asyncio.Queue`. If the queue is full → `STTQueueFull` → **HTTP 429**.
3. The single `_worker_loop` task takes one job at a time and runs
   `_infer` in a thread, so inference is fully serialized.
4. If the caller's future was cancelled (client disconnected), the worker
   **skips** the job — no inference is wasted.
5. On success the future resolves with the result dict; on failure the future
   gets the exception and `failed_requests` is incremented. The worker never
   dies (exceptions are caught per job).

Language resolution (`_infer`):

- `/transcribe-pcm` and `/transcribe`: `job.language or WHISPER_LANGUAGE`
  (i.e. request query param wins; otherwise the configured default `hi`).
- `/v1/transcriptions`: `use_config_default_language=False`, so the config
  default is **never** forced — omitted `language` means true auto-detection
  and the response reports the detected language.

Confidence is derived from the mean segment log-probability as
`exp(mean(avg_logprob))`, clamped to `[0, 1]` and rounded to 4 decimals.

---

## Endpoints

Base URL: `http://localhost:8000`.

| Method | Path | Envelope? | Auth | Purpose |
|---|---|---|---|---|
| `GET` | `/` | no | none | Service discovery / endpoint list. |
| `POST` | `/transcribe-pcm` | yes | none | Raw PCM utterance (LiveKit path). |
| `POST` | `/transcribe` | yes | none | Multipart audio file upload. |
| `POST` | `/v1/transcriptions` | no | optional bearer | URL → timestamped segments. |
| `GET` | `/v1/stt/health` | no | none | Liveness + model/queue info. |

### GET /

Not shown in the OpenAPI schema (`include_in_schema=False`). Returns the
service name, version and available endpoints:

```json
{
  "service": "stt-whisper",
  "version": "1.0.0",
  "endpoints": [
    "/transcribe-pcm",
    "/transcribe",
    "/v1/transcriptions",
    "/v1/stt/health"
  ]
}
```

---

### POST /transcribe-pcm (LiveKit client path)

Transcribes a **raw mono `s16le` PCM** body (no WAV header).

**Headers**

| Header | Default | Description |
|---|---|---|
| `X-Sample-Rate` | `24000` | Sample rate of the PCM body. Must be in `[8000, 96000]`. |
| `Content-Type` | — | Use `application/octet-stream`. |

**Query parameters** (all optional)

| Name | Type | Default | Description |
|---|---|---|---|
| `language` | string | configured `WHISPER_LANGUAGE` (`hi`) | Language hint, e.g. `hi`, `en`. |
| `request_id` | string | *(none)* | Correlation ID echoed in logs. |
| `word_timestamps` | bool | `false` | Include per-word timestamps in `data.words`. |
| `initial_prompt` | string | *(none)* | Domain-specific vocabulary hint for Whisper. Each app can pass its own prompt per request. |

**Validation** (all → `400`)

- Empty body.
- Odd number of bytes (must be whole `int16` samples).
- Sample rate outside `8000–96000`.
- Duration shorter than `WHISPER_MIN_UTTERANCE_MS` (default 40 ms).

**Audio handling**

- Bytes → `int16` → `float32 / 32768.0`.
- Resampled to `WHISPER_TARGET_SAMPLE_RATE` (16 kHz) with
  `scipy.signal.resample_poly` only when the input rate differs.
- Optionally dumped to a WAV when `WHISPER_AUDIO_DUMP_DIR` is set. The dump is
  written at `WHISPER_TARGET_SAMPLE_RATE` (16 kHz), i.e. the resampled audio
  the model actually sees, so the conversion is directly verifiable.

**Example**

```bash
curl -X POST http://localhost:8000/transcribe-pcm \
  -H "X-Sample-Rate: 48000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @utterance.pcm
```

With domain vocabulary:

```bash
curl -X POST "http://localhost:8000/transcribe-pcm?initial_prompt=gehu%20mein%20pila%20rog" \
  -H "X-Sample-Rate: 48000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @utterance.pcm
```

```json
{
  "success": true,
  "message": "PCM transcription completed successfully",
  "data": {
    "transcript": "गेहूं में पीला रोग लग गया है",
    "language": "hi",
    "confidence": 0.92
  }
}
```

**Silence is not an error**: silence yields `200` with
`data.transcript: ""` (the agent treats this as "no speech"). Note the model
may still return a small `confidence` for such clips.

---

### POST /transcribe (offline file upload)

Offline / batch transcription from an uploaded audio file. Accepts
WAV/MP3/FLAC/OGG/M4A (anything `faster_whisper.audio.decode_audio` can decode).

**Multipart form field**

| Field | Required | Description |
|---|---|---|
| `file` | yes | Audio file. |

**Query parameters** — `language`, `request_id`,
`word_timestamps` (default `false`).

**Example**

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@recording.wav" \
  -F "language=hi"
```

Response uses the same `ApiResponse` envelope, with
`message: "Transcription completed successfully"` and the same `data` shape as
`/transcribe-pcm`. Empty file → `400`. Undecodable audio → `400`.

---

### POST /v1/transcriptions (audio URL → timestamped segments)

Downloads audio from a URL, transcribes it, and returns **segment-level** (and
optionally **word-level**) timestamps. This endpoint returns the JSON body
**directly** — no `{success, message, data}` envelope.

**Auth** — enforced only when `STT_API_TOKEN` is set:

```
Authorization: Bearer <STT_API_TOKEN>
```

Missing/invalid token → `401` with `WWW-Authenticate: Bearer`. When
`STT_API_TOKEN` is unset, auth is disabled (dev).

**Request body** (`application/json`)

| Field | Type | Default | Rules |
|---|---|---|---|
| `audioUrl` | string | — (required) | `http`/`https` only; length 8–2048. |
| `language` | string \| null | `null` | ISO code or name, max 32 chars. `null`/omitted → auto-detect. |
| `wordTimestamps` | bool | `false` | Add `words: [{text, start, end}]` to each segment. |

**Download constraints** (`app/services/audio_fetch.py`)

- Only `http` / `https` schemes; anything else → `400` before any network call.
- Redirects are followed.
- `Content-Length` is checked up front when present; the stream is also counted
  and aborted if it exceeds `STT_MAX_DOWNLOAD_MB` (default 100 MB).
- Total timeout `STT_DOWNLOAD_TIMEOUT_S` (default 120 s).
- Non-200 remote status, download error, or empty body → `400`.
- Downloaded bytes are decoded to 16 kHz `float32` via faster-whisper's
  decoder; empty/undecodable → `400`.

**Example**

```bash
curl -X POST http://localhost:8000/v1/transcriptions \
  -H "Authorization: Bearer $STT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "audioUrl": "https://storage.mssplonline.in/e-learning/materials/content_library/audios/853fe5e7-46bf-4c5b-b117-64bd9c3f5c56.wav",
    "language": "en",
    "wordTimestamps": true
  }'
```

**Response** (`response_model_exclude_none=True`, so `words` is omitted unless
requested):

```json
{
  "language": "en",
  "durationMs": 268301,
  "segments": [
    {
      "start": 0.0,
      "end": 4.32,
      "text": "Welcome to today's class.",
      "words": [
        { "text": "Welcome", "start": 0.0, "end": 0.61 },
        { "text": "to", "start": 0.61, "end": 0.8 }
      ]
    },
    { "start": 4.32, "end": 9.17, "text": "Today we are going to discuss..." }
  ]
}
```

- `language` is the language actually used (request hint or detected).
- `durationMs` is the audio duration reported by Whisper, rounded to ms.
- Segment `start`/`end` are rounded to 2 decimals; word times to 3 decimals.
- This endpoint does **not** apply the configured default language or the
  domain `initial_prompt` (auto-detect / neutral behavior).

**Status codes**

| Status | Condition |
|---|---|
| 200 | Success. |
| 400 | Invalid URL/scheme, download failed, empty/undecodable audio. |
| 401 | Missing/invalid bearer token (when `STT_API_TOKEN` is set). |
| 422 | Request body fails schema validation (e.g. missing `audioUrl`). |
| 429 | Inference queue full (retryable). |
| 500 | Transcription failure. |

---

### GET /v1/stt/health

Liveness + model/device/queue/counter info. No auth.

```bash
curl http://localhost:8000/v1/stt/health
```

```json
{
  "status": "ok",
  "model": "large-v3",
  "model_loaded": true,
  "device": "cpu",
  "compute_type": "int8",
  "language": "hi",
  "target_sample_rate": 16000,
  "queue": { "depth": 0, "maxsize": 32 },
  "requests": { "total": 42, "failed": 1 }
}
```

`requests.total` counts all inference jobs attempted; `requests.failed` counts
jobs that raised.

---

## Response schemas

`ApiResponse` (`app/schemas/response.py`) — used by `/transcribe-pcm` and
`/transcribe`:

```python
class ApiResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict[str, Any] = {}
```

The `data` object for PCM/file endpoints:

```jsonc
{
  "transcript": "full text, segments joined with a space",
  "language": "used or detected language code",
  "confidence": 0.0,        // exp(mean(avg_logprob)), clamped [0,1], 4 dp
  "words": [                 // present only when word_timestamps=true
    { "text": "...", "start": 0.0, "end": 0.5 }
  ]
}
```

`/v1/transcriptions` schemas (`app/schemas/transcription.py`):

```python
class TranscriptionRequest(BaseModel):
    audioUrl: str            # 8..2048, http/https
    language: str | None = None
    wordTimestamps: bool = False

class WordOut(BaseModel):
    text: str; start: float; end: float

class SegmentOut(BaseModel):
    start: float; end: float; text: str
    words: list[WordOut] | None = None

class TranscriptionResponse(BaseModel):
    language: str
    durationMs: int
    segments: list[SegmentOut]
```

---

## Error handling

| Status | Where | Condition |
|---|---|---|
| 400 | `/transcribe-pcm` | Empty / odd-length PCM, invalid sample rate, utterance too short, invalid language code. |
| 400 | `/transcribe` | Empty or undecodable file, invalid language code. |
| 400 | `/v1/transcriptions` | Bad URL scheme, download failure, size-limit exceeded, empty/undecodable audio, invalid language code. |
| 401 | `/v1/transcriptions` | Missing/invalid bearer token (only when `STT_API_TOKEN` set). |
| 422 | any JSON body | Schema validation failure (FastAPI/Pydantic). |
| 429 | inference endpoints | Inference queue full (`WHISPER_QUEUE_MAXSIZE`). Retryable. |
| 500 | inference endpoints | Unexpected transcription failure. Retryable. |
| 200 + empty transcript | `/transcribe-pcm`, `/transcribe` | Silence / no speech — handled gracefully, **not** an error. |

Client disconnects propagate `asyncio.CancelledError`; the job's future is
cancelled and the worker skips it.

---

## Design notes

- **Audio**: PCM is converted to `float32 / 32768.0` and resampled to 16 kHz
  (`scipy.signal.resample_poly`). File/URL audio is decoded by
  `faster_whisper.audio.decode_audio`. No WAV container is assumed on the PCM
  path.
- **Latency**: the model stays in memory; defaults are `beam_size=5`,
  `temperature=0.0`, `vad_filter=False` (the agent already VAD-segments).
- **Hallucination guarding**: two layers. (1) **Signal check**: if the decoded
  audio peak amplitude is below `WHISPER_SILENCE_LEVEL`, inference is
  short-circuited and an empty transcript is returned — this catches full
  digital silence, which Whisper often "hallucinates" on
  (`no_speech_prob ≈ 0.000`). (2) **Segment filter**: every decoded segment is
  dropped when its text is empty, when its `compression_ratio` exceeds
  `WHISPER_COMPRESSION_RATIO_THRESHOLD` (repetition loops), or when
  `no_speech_prob > WHISPER_NO_SPEECH_THRESHOLD` **and** `avg_logprob <
  WHISPER_LOGPROB_THRESHOLD` (silence/noise). Both layers are
  decoding-agnostic: they keep `vad_filter=False` and `temperature=0.0` for the
  `/transcribe-pcm` path. Dropped silence yields an empty transcript (still
  `200`, treated as "no speech").
- **Concurrency**: all inference is serialized through a single asyncio worker
  + bounded queue (`WHISPER_QUEUE_MAXSIZE`). Concurrent requests are queued,
  never dropped; a full queue returns `429`.
- **Cancellation**: a client disconnect cancels its future; the worker skips
  cancelled jobs so no GPU inference is wasted.
- **Device/compute**: `WHISPER_DEVICE=auto` probes CTranslate2 for CUDA; on
  CPU, `float16` is coerced to `int8`.
- **Debugging**: set `WHISPER_AUDIO_DUMP_DIR` to write every PCM request to a
  WAV. Each request logs `request_id`, sample rate, duration and transcript.
- <a id="domain-initial-prompt"></a>**Domain initial prompt**: a Hindi
  agriculture vocabulary string is defined as `DEFAULT_INITIAL_PROMPT`
  (overridable via `WHISPER_INITIAL_PROMPT`). The `/transcribe-pcm` endpoint
  accepts an optional `initial_prompt` query parameter so each app can pass its
  own vocabulary hint per request. If omitted, no domain prompt is applied.
- **macOS warning filter**: numpy's Accelerate BLAS emits spurious
  `RuntimeWarning: ... encountered in matmul` messages during the mel-spectrogram
  matmul; `whisper_service.py` filters exactly that message so real warnings
  still surface.

---

## Testing

Tests use `pytest`, `fastapi.testclient.TestClient`, and monkeypatch the model
so **no real inference or download happens**.

```bash
pytest -q
```

Coverage highlights:

- `tests/test_api.py` — success envelope, default sample-rate header, language
  hint passthrough, `400`s for empty/odd/short/bad-rate bodies, file upload,
  health, and that 5 concurrent requests are serialized (queue depth > 0) rather
  than run in parallel.
- `tests/test_transcriptions.py` — segment timestamps, optional word
  timestamps, auto-detected language when omitted (proves the config default is
  not forced), invalid scheme → `400`, download failure → `400`, bearer auth
  enabled/disabled, missing `audioUrl` → `422`, health.

---

## Interactive API docs

FastAPI generates live docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

The app metadata is:

- Title: `STT Whisper Server`
- Version: `1.0.0`
