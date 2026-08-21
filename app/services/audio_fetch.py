import logging
from collections.abc import AsyncIterator

import httpx

from ..config import settings

logger = logging.getLogger("stt.fetch")

_CHUNK = 1 << 16

ALLOWED_SCHEMES = {"http", "https"}


class AudioDownloadError(Exception):
    """Audio could not be fetched from the remote URL -> HTTP 400."""


async def _stream(url: str, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    max_bytes = settings.max_download_mb * 1024 * 1024
    received = 0
    async with client.stream("GET", url) as resp:
        if resp.status_code != 200:
            raise AudioDownloadError(
                f"Remote server returned HTTP {resp.status_code} for the audio URL"
            )
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise AudioDownloadError(
                f"Audio larger than the {settings.max_download_mb}MB limit"
            )
        async for chunk in resp.aiter_bytes(_CHUNK):
            received += len(chunk)
            if received > max_bytes:
                raise AudioDownloadError(
                    f"Audio larger than the {settings.max_download_mb}MB limit"
                )
            yield chunk


async def fetch_audio(url: str) -> bytes:
    """Download an audio file, enforcing scheme and size limits."""
    if (scheme := url.split(":", 1)[0].lower()) not in ALLOWED_SCHEMES:
        raise AudioDownloadError(f"Unsupported URL scheme: {scheme!r}")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(settings.download_timeout_s),
        ) as client:
            parts = [chunk async for chunk in _stream(url, client)]
    except httpx.HTTPError as exc:
        logger.warning("Audio download failed url=%s: %s", url, exc)
        raise AudioDownloadError("Could not download audio from the given URL") from exc

    content = b"".join(parts)
    if not content:
        raise AudioDownloadError("Downloaded audio is empty")
    return content
