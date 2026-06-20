# Vertex AI Integration

This project includes a native integration for Google Cloud Vertex AI to provide Gemini model support via Application Default Credentials (ADC). 

## Architecture

The integration is built as a custom provider (`VertexGeminiProvider`) extending `OpenAIChatTransport`. Vertex AI exposes an OpenAI-compatible endpoint, but it has several quirks that this proxy resolves:

1. **Authentication**: Claude Code and the proxy normally expect API keys. Vertex AI requires short-lived OAuth 2.0 access tokens.
    - We use `google-auth` to fetch credentials via ADC (e.g. `GOOGLE_APPLICATION_CREDENTIALS`).
    - The provider is registered with `static_credential="used_via_adc"` in `config/provider_catalog.py` to bypass the proxy's strict API key validation.

2. **Model Discovery**: Vertex AI's OpenAI-compatible endpoint does *not* support the `/models` discovery path (it returns 404).
    - We hardcode the supported model list in `list_model_ids()`.
    - Currently restricted to: `claude-3-5-vertex-gemini-3.1-pro` and `claude-3-5-vertex-gemini-3.1-flash-lite`.

3. **Claude Code Fallback Bypass**: Claude Code contains an internal, obfuscated list of "recognized" models. If it encounters a raw `google/gemini-...` string, it panics, prepends `anthropic/`, and attempts to validate the proxy token directly against Anthropic's production servers, causing a 401 Unauthorized loop.
    - **Solution**: We alias the models using a `claude-3-5-*` prefix so Claude Code natively trusts them.
    - The `_build_request_body` interceptor translates these aliases back to the real Google models (e.g., `google/gemini-3.1-pro-preview`) before sending the payload.

4. **Subagent Compatibility (Reasoning Effort)**: When Claude Code spawns subagents, it explicitly disables thinking. The proxy's Gemini base class translates this to `reasoning_effort="none"`.
    - Google AI Studio accepts `"none"`, but Vertex AI strictly rejects it with HTTP 400.
    - We patch `_build_request_body` to delete the `reasoning_effort` key if its value is `"none"`.

5. **Region and Endpoint**: Gemini 3.1 preview models are exclusively available on the `global` region in Vertex AI. The URL builder dynamically switches the hostname to `aiplatform.googleapis.com` when the region is `global`, avoiding the invalid `global-aiplatform` prefix.

## Usage

Set the following in `.env`:
```env
VERTEX_PROJECT=your-project-id
VERTEX_REGION=global  # Required for Gemini 3.1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
MODEL=vertex_gemini/claude-3-5-vertex-gemini-3.1-pro
```
