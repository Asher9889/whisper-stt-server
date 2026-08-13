# STT Whisper Server

Self-hosted **faster-whisper** STT server exposing the exact HTTP contract the
LiveKit voice agent expects: one VAD-segmented utterance per request, raw
mono `s16le` PCM, `{success, message, data}` response envelope.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Server listens on `0.0.0.0:8000` (configurable, see `.env.example`).

The model is downloaded on first start (HuggingFace) and kept resident in
memory. Set `WHISPER_MODEL` to `small`, `medium`, `large-v3`, or `turbo`
(default). On a CPU-only machine `float16` automatically falls back to `int8`.

## Endpoints

### POST /transcribe-pcm  (LiveKit client path)

Raw mono `s16le` PCM in the body, sample rate in the `X-Sample-Rate` header
(defaults to `48000`).

```bash
curl -X POST http://localhost:8000/transcribe-pcm \
  -H "X-Sample-Rate: 48000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @utterance.pcm
```

Optional query params: `language` (`hi`), `request_id`, `word_timestamps`.

Response:

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

Silence yields `200` with `data.transcript: ""` (the agent treats it as "no
speech"); it is never an error.

### POST /transcribe  (offline file upload)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@recording.wav" \
  -F "language=hi"
```

### GET /v1/stt/health

Liveness + model/device info + queue depth:

```bash
curl http://localhost:8000/v1/stt/health
```

## Error handling

| Status | Condition |
|---|---|
| 400 | Empty / too-short / odd-length PCM body, invalid sample rate |
| 429 | Inference queue full (retryable) |
| 500 | Transcription failure (retryable) |
| 200 + empty transcript | Silence / no speech (ignored gracefully) |

## Design notes

- **Audio**: PCM is converted to `float32 / 32768.0` and resampled to 16 kHz
  (`scipy.signal.resample_poly`) before transcription. No WAV container.
- **Latency**: model kept in memory; `beam_size=5`, `temperature=0.0`,
  `vad_filter=False` (agent already VAD-segments), agriculture `initial_prompt`.
- **Concurrency**: all inference is serialized through a single asyncio worker
  + queue (`WHISPER_QUEUE_MAXSIZE`); concurrent requests are queued, never
  dropped. Full queue → 429.
- **Cancellation**: a client disconnect cancels its future; the worker skips
  cancelled jobs so no GPU inference is wasted.
- **Debugging**: set `WHISPER_AUDIO_DUMP_DIR` to write every PCM request to a
  WAV; each request logs `request_id`, sample rate, duration and transcript.
