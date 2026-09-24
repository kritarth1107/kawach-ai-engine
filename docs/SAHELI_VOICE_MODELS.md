# Saheli voice + model notes (Phase 1)

## Companion / briefs LLM
- Model: `gemini-3.5-pro` via Vertex (`VERTEX_CHAT_MODEL`, `VERTEX_CAREGIVER_CHAT_MODEL`)
- Location: `VERTEX_LOCATION` (defaults to `global` for Pro-class; embeddings still use `GCP_REGION`, usually `asia-south1`)
- Probe (2026-09-24, project `kavach-care`): `gemini-3.5-pro` returned 404 in asia-south1 / us-central1 / global (allowlist/preview). `gemini-3.5-flash` works in asia-south1 + global. Closest working Pro: `gemini-3.1-pro-preview` @ global.

## STT
- Prefer Speech-to-Text V2 `chirp_3` (`SPEECH_LOCATION=us` or `eu`)
- Fallback: Gemini 3.5 audio understanding via Vertex when Chirp fails / unavailable

## TTS
- ElevenLabs when `ELEVENLABS_API_KEY` or `ELEVEN_LABS_API_KEY` is set
- Optional: `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID` (default `eleven_multilingual_v2`)
- If key missing: text-only WhatsApp reply + log skip
