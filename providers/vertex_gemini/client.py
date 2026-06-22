"""Vertex AI Gemini provider."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import google.auth
import google.auth.transport.requests

from providers.base import ProviderConfig
from providers.exceptions import AuthenticationError
from providers.gemini.request import build_request_body
from providers.transports.openai_chat import OpenAIChatTransport

VERTEX_BASE = (
    "https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project}"
    "/locations/{region}/endpoints/openapi"
)


class VertexGeminiProvider(OpenAIChatTransport):
    """Gemini via Vertex AI."""

    def __init__(self, config: ProviderConfig, settings=None):
        from dotenv import load_dotenv

        load_dotenv()

        if settings is not None:
            project = settings.vertex_project
            region = settings.vertex_region
        else:
            project = os.environ.get("VERTEX_PROJECT", "")
            region = os.environ.get("VERTEX_REGION", "us-central1")

        if not project:
            raise AuthenticationError("VERTEX_PROJECT is not configured in .env")

        hostname = (
            "aiplatform.googleapis.com"
            if region == "global"
            else f"{region}-aiplatform.googleapis.com"
        )
        base_url = f"https://{hostname}/v1beta1/projects/{project}/locations/{region}/endpoints/openapi"

        try:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            req = google.auth.transport.requests.Request()
            creds.refresh(req)
        except Exception as e:
            raise AuthenticationError(
                f"Failed to authenticate with Google Cloud: {e}"
            ) from e

        super().__init__(
            config,
            provider_name="VERTEX_GEMINI",
            base_url=config.base_url or base_url,
            api_key=creds.token,
        )
        self._creds = creds
        self._auth_req = req

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        if not self._creds.valid or self._creds.expired:
            await asyncio.to_thread(self._creds.refresh, self._auth_req)
            self._client.api_key = self._creds.token

        return await super()._create_stream(body)

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset(
            [
                "claude-opus-4-8[1m]",
                "claude-sonnet-4-6[1m]",
            ]
        )

    def _build_request_body(self, request, thinking_enabled=None):
        request_copy = request.copy() if hasattr(request, "copy") else request
        model_name = getattr(request_copy, "model", "")
        if isinstance(model_name, str):
            if "claude-opus" in model_name:
                request_copy.model = "google/gemini-3.1-pro-preview"
            elif "claude-sonnet" in model_name:
                request_copy.model = "google/gemini-3.1-flash-lite"

        body = build_request_body(
            request_copy,
            thinking_enabled=self._is_thinking_enabled(request_copy, thinking_enabled),
            tool_call_extra_content_by_id={},
        )
        if body.get("reasoning_effort") == "none":
            del body["reasoning_effort"]

        if body.get("model") == "claude-opus-4-8[1m]":
            body["model"] = "google/gemini-3.1-pro-preview"
        elif body.get("model") == "claude-sonnet-4-6[1m]":
            body["model"] = "google/gemini-3.1-flash-lite"

        return body
