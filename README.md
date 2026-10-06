---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Agent capabilities and requirements

- **Web links:** Uses `web_search` to find sources and `visit_webpage` to read
  pages. Direct PDF and EPUB links can be read with `read_book`.
- **YouTube:** `youtube_transcript` reads available captions/transcripts. It
  does not require captions to inspect visuals: questions containing YouTube
  links trigger up to 12 frames sampled across the video, passed to the vision
  model. The frame sampling can miss brief events; the agent can also use
  captions when available.
- **Books and documents:** Attached PDF and EPUB files are extracted to text.
  Attached `.txt`, `.md`, and `.py` files are also read. Extracted content is
  limited to 100,000 characters, and the app marks when text is truncated.
- **Images:** Common image attachments are passed directly to the model, but
  only when the configured model actually accepts images. The app asks Ollama
  for the model's capabilities and, for a text-only model, skips image
  attachments and YouTube frame sampling instead of spending minutes producing
  input the model cannot read. Override the probe with `LLM_VISION=true|false`.
- **Context window:** Attachment extracts are capped at 100,000 characters, but
  Ollama's OpenAI-compatible endpoint takes no per-request context size, so the
  server default (4096) applies and over-long prompts are truncated server-side
  without warning. Raise it with `OLLAMA_CONTEXT_LENGTH` before `ollama serve`.
  Pick the value to fit your VRAM: on an 8 GB card an 8B model at Q4 only stays
  fully on the GPU at 4096, and every step above that spills layers to the CPU
  (measured on an RX 6600: 4096 → 39.6 tok/s fully on GPU, 8192 → 30.1 tok/s at
  27% CPU, 16384 → 8.8 tok/s at 37% CPU). Check with `ollama ps`, which prints
  the CPU/GPU split. `max_tokens` must also fit inside the window alongside the
  ~1,200-token system prompt, or the very first call cannot succeed.
- **Calculations and code:** `python_interpreter` is available for arithmetic,
  table reductions, and running attached Python source.
- **Wikipedia:** `wikipedia(query=...)` searches English Wikipedia and returns
  the article's plain text, which avoids spending agent steps triaging search
  results. Note that it requests `explaintext`, which flattens tables -- so it
  is poor for questions that depend on reading a specific table row.
- **Answer formatting:** The scoring API compares answers exactly, so the app
  normalizes the model's reply before submitting: it unwraps code fences, drops a
  leading "the answer is", strips markdown emphasis and surrounding quotes, and
  removes a single trailing period.
- **Ollama tool calls:** The app uses automatic tool selection, so the model is
  not forced to call a tool when it is ready to answer.
- **Thinking mode cannot be disabled over `/v1`.** The app sends
  `extra_body={"think": False}`, but Ollama's OpenAI-compatible endpoint ignores
  the `think` parameter entirely; only the native `/api/chat` endpoint honours
  it. For a model whose thinking cannot be switched off, reasoning tokens count
  against `max_tokens` and can consume the whole budget before any answer or
  tool call is emitted -- `qwen3-vl:8b` reliably produces empty content this
  way. Prefer a model with no thinking mode (for example `qwen2.5:7b`, the
  default) when running against Ollama.
- **Answers must come through `final_answer`.** smolagents rejects a plain-text
  reply with "model output does not contain any JSON blob", costing a step; a
  model that keeps doing this exhausts `max_steps` and returns narration instead
  of an answer. The instructions therefore lead with that rule, and are kept
  short deliberately: a long system prompt buries it and small models stop
  complying.
- **Audio:** Common attached audio formats and direct audio URLs are
  transcribed with the Hugging Face Inference Provider model
  `openai/whisper-large-v3-turbo` by default. Set `HF_TOKEN` to a Hugging Face
  token with Inference Providers access. You can set
  `AUDIO_TRANSCRIPTION_MODEL` to select another supported transcription model.
  Supported file extensions include `.mp3`, `.mpga`, `.wav`, `.m4a`, `.aac`,
  `.flac`, `.ogg`, `.webm`, `.aif`, and `.aiff`.
- **Excel:** `.xlsx` and `.xlsm` attachments are read sheet-by-sheet and
  supplied as CSV text. Workbooks larger than 100,000 characters are rejected
  rather than silently truncated.

## Choosing a model provider

The agent talks to any OpenAI-compatible chat-completions endpoint. Pick one
with `LLM_PROVIDER`; the default is `ollama`, so an existing local setup keeps
working unchanged.

| `LLM_PROVIDER` | endpoint | key | default model |
| --- | --- | --- | --- |
| `ollama` (default) | `http://localhost:11434/v1` | none | `qwen2.5:7b` |
| `gemini` | Google AI Studio | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| `groq` | Groq | `GROQ_API_KEY` | `qwen/qwen3.8-27b` |
| `huggingface` | HF Inference Providers | `HF_TOKEN` | `Qwen/Qwen2.5-72B-Instruct` |
| `openai` | OpenAI | `OPENAI_API_KEY` | `gpt-4.1-mini` |

`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MAX_TOKENS` override any
single field. A missing key fails at startup with a message naming the variable
to set.

```powershell
# Hosted run, e.g. Gemini (get a free key at https://aistudio.google.com/apikey)
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "your-key"   # never commit this
python app.py
```

A local 7-8B model is unlikely to reach the 30% pass mark: GAIA Level 1 needs
multi-hop research and exact-match answers, and published passes generally use
a hosted model or a ~70B local one. Local running is good for development; use
a hosted provider for a scoring run.

## Run locally with a Python venv

Use a local virtual environment so the app uses the same packages as the project.

```powershell
# 1) Create and activate the environment
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Install/update Ollama to 0.12.7 or newer, then pull the model (first time only)
ollama pull qwen3-vl:8b
$env:OLLAMA_HOST = "0.0.0.0:11434"
# The Ollama default context (4096) silently truncates long prompts, but a
# larger window pushes an 8B model off an 8 GB GPU. 8192 is a reasonable
# compromise; check the split afterwards with `ollama ps`.
$env:OLLAMA_CONTEXT_LENGTH = "8192"
ollama serve
```

In a second PowerShell terminal, run the app:

```powershell
. .\.venv\Scripts\Activate.ps1
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
$env:OLLAMA_MODEL = "qwen3-vl:8b"
$env:PORT = "7860"
# Needed only if a task includes audio to transcribe; do not commit this token.
$env:HF_TOKEN = "hf_your_token"
python app.py
```

Open [http://localhost:7860](http://localhost:7860) and enter your Hugging Face
username. The local app defaults to the Ollama URL `http://localhost:11434/v1`.

## Run with Docker

The Docker app connects to Ollama running on your Windows host. Pull the model
and configure Ollama to accept connections from Docker:

```powershell
ollama pull qwen3-vl:8b
$env:OLLAMA_HOST = "0.0.0.0:11434"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
ollama serve
```

If Ollama is already running in the background, restart it after setting
`OLLAMA_HOST` and `OLLAMA_CONTEXT_LENGTH`. In another PowerShell window, build and run the app:

```powershell
docker build -t agents-course-app .
# Needed only if a task includes audio to transcribe; do not commit this token.
$env:HF_TOKEN = "hf_your_token"
docker run --rm -it -p 7860:7860 -e HF_TOKEN=$env:HF_TOKEN agents-course-app
```

Open [http://localhost:7860](http://localhost:7860) and enter your Hugging Face
username. Docker connects to Ollama at `host.docker.internal:11434`. For a
non-Docker run, the app defaults to `http://localhost:11434/v1`. Set
`OLLAMA_BASE_URL` or `OLLAMA_MODEL` to override the connection or model.

The scoring API may return HTTP 404 for a task whose metadata contains a file
name but has no attachment mapped to that task ID. The app now reports this
clearly, continues with the question and other available evidence, and still
submits an answer rather than silently dropping the task. No code can read
attachment bytes the scoring API does not provide.