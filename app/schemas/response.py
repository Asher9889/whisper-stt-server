from typing import Any

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    """Envelope returned by every STT endpoint.

    Matches ``app/schemas/response.py::ApiResponse`` in the agent repo. The
    LiveKit clients parse ``payload["data"]`` and read ``data.transcript``,
    ``data.language`` and ``data.confidence``.
    """

    success: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
