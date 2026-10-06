import json
import os
import posixpath
import re
import subprocess
import tempfile
import time
import zipfile
from io import BytesIO
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

import gradio as gr
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from smolagents import (
    DuckDuckGoSearchTool,
    OpenAIModel,
    PythonInterpreterTool,
    Tool,
    ToolCallingAgent,
    VisitWebpageTool,
)

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
MAX_VIDEO_BYTES = 100_000_000
MAX_DOCUMENT_BYTES = 25_000_000
MAX_DOCUMENT_CHARS = 100_000

ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:\-]?\s*",
    re.IGNORECASE,
)

# Conversational lead-ins that end in a colon, e.g.
# "Sure, here are the vegetables from your list, sorted alphabetically: a, b, c"
ANSWER_LEAD_IN_PATTERN = re.compile(
    r"^(?:sure|certainly|of\s+course|okay|ok|alright|absolutely|yes)\b[^:]{0,120}:\s*"
    r"|^(?:here\s+(?:are|is)|the\s+(?:vegetables|list|items|results?)\b)[^:]{0,120}:\s*",
    re.IGNORECASE,
)

# Chatty tails the models append after the actual value, e.g.
# "... zucchini You can use these to create your categories"
TRAILING_COMMENTARY_PATTERN = re.compile(
    r"\s+(?:you\s+can|you\s+may|you\s+might|let\s+me\s+know|i\s+hope|hope\s+(?:this|that)|"
    r"if\s+you|feel\s+free|note\s+that|please\s+note|these\s+(?:are|can|should)|"
    r"this\s+(?:is|list|gives))\b.*$",
    re.IGNORECASE | re.DOTALL,
)


# Expanded only when the question itself forbids abbreviations, e.g. "Just give
# me the city name without abbreviations" -- the model answers "St. Petersburg"
# and loses an exact match it had otherwise earned.
ABBREVIATION_EXPANSIONS = {
    "st": "Saint",
    "ste": "Sainte",
    "mt": "Mount",
    "ft": "Fort",
    "pt": "Point",
}
ABBREVIATION_PATTERN = re.compile(
    r"\b(" + "|".join(ABBREVIATION_EXPANSIONS) + r")\.", re.IGNORECASE
)


def _expand_abbreviations(text: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        return ABBREVIATION_EXPANSIONS[match.group(1).lower()]

    return ABBREVIATION_PATTERN.sub(replace, text)


def _normalize_answer(answer: str, question: str = "") -> str:
    """Reduce a model reply to the bare value the scoring API compares exactly."""
    text = answer.strip()

    fenced = re.fullmatch(r"```[a-zA-Z0-9_+-]*\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\*\*|__|\*|`)", "", text).strip()

    # Peel wrappers in order: "Sure, here are ...:" then "The answer is ...".
    text = ANSWER_LEAD_IN_PATTERN.sub("", text, count=1).strip()
    text = ANSWER_PREFIX_PATTERN.sub("", text, count=1).strip()
    text = TRAILING_COMMENTARY_PATTERN.sub("", text, count=1).strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()

    if "abbreviation" in question.lower():
        text = _expand_abbreviations(text)

    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].rstrip()

    return text or answer.strip()


# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
# Every provider below speaks the OpenAI chat-completions protocol, so the only
# things that change are the base URL, the key, and a sensible default model.
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        "default_key": "ollama",
        "model": "qwen2.5:7b",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "qwen/qwen3.8-27b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
    },
}


def resolve_llm_config() -> dict:
    """Pick the provider, model, endpoint and key from the environment.

    `LLM_PROVIDER` selects a provider; `LLM_MODEL`, `LLM_BASE_URL` and
    `LLM_API_KEY` override any individual field. The legacy `OLLAMA_*` names
    still work so existing local setups keep running unchanged.
    """
    name = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER {name!r}. Choose one of: {', '.join(sorted(PROVIDERS))}."
        )
    provider = PROVIDERS[name]

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv(provider["key_env"])
        or provider.get("default_key")
    )
    if not api_key:
        raise RuntimeError(
            f"Provider {name!r} needs an API key: set {provider['key_env']} "
            "(or LLM_API_KEY) in the environment."
        )

    if name == "ollama":
        model_id = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or provider["model"]
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or provider["base_url"]
    else:
        model_id = os.getenv("LLM_MODEL") or provider["model"]
        base_url = os.getenv("LLM_BASE_URL") or provider["base_url"]

    max_tokens = int(
        os.getenv("LLM_MAX_TOKENS") or os.getenv("OLLAMA_MAX_TOKENS") or "2000"
    )
    return {
        "provider": name,
        "model_id": model_id,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
    }


class BasicAgent:
    def __init__(self):
        search_tool = DuckDuckGoSearchTool()
        config = resolve_llm_config()
        print(
            f"LLM provider: {config['provider']} | model: {config['model_id']} | "
            f"max_tokens: {config['max_tokens']}"
        )
        model_kwargs = {}
        if config["provider"] == "ollama":
            # Ollama-only passthrough; other providers reject unknown fields.
            model_kwargs["extra_body"] = {"think": False}
        model = OpenAIModel(
            model_id=config["model_id"],
            api_base=config["base_url"],
            api_key=config["api_key"],
            max_tokens=config["max_tokens"],
            tool_choice="auto",
            **model_kwargs,
        )
        self.agent = ToolCallingAgent(
            tools=[
                search_tool,
                WikipediaTool(),
                PoliteVisitWebpageTool(),
                YouTubeTranscriptTool(),
                ReadBookTool(),
                TranscribeAudioTool(),
                PythonInterpreterTool(),
            ],
            model=model,
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "16")),
            instructions=(
                "ANSWERING: finish by calling the tool whose name is exactly `final_answer` -- that "
                "exact name, never `answer` or any other variation -- and pass your result as its "
                "`answer` argument, i.e. {\"name\": \"final_answer\", \"arguments\": {\"answer\": \"...\"}}. "
                "Calling a tool that is not in the provided tool list aborts the run. A plain-text reply "
                "is discarded and wastes a step. Call it as soon as you have enough information, and "
                "never repeat a tool call you have already made with the same arguments.\n\n"
                "RESEARCH: for any encyclopedic fact -- people, places, works, events, awards, "
                "competitions, sports results, species -- call `wikipedia` FIRST, before `web_search`. "
                "Call it with `query` to list the article's sections, then again with `section=` to read "
                "the one you need; its tables are clean and compact. Only fall back to `web_search` and "
                "`visit_webpage` when Wikipedia genuinely lacks the fact. Whole web pages cost many times "
                "more context than a Wikipedia section and will exhaust the rate limit.\n"
                "Use `python_interpreter` for ANY deterministic work -- reversing or "
                "transforming text, counting, sorting, scanning a table, set operations, arithmetic -- "
                "rather than working it out in your head. Auto-generated transcripts have no punctuation "
                "or speaker labels: a reply to a question is simply the words that follow it. If an "
                "attachment is reported unavailable, say so and answer from other evidence.\n\n"
                "FORMAT (graded by exact string match): pass ONLY the bare value to `final_answer` -- no "
                "preamble, no explanation, no markdown, no trailing commentary.\n"
                "Correct: `3` | `b, e` | `FunkMonk` | `Extremely`\n"
                "Wrong: `The answer is 3.` | `Sure, here are the vegetables: ...`\n"
                "Match the granularity asked for: 'first name' -> `Claus`, not `Claus Peter`; 'IOC code' "
                "-> `CUB`, not `Cuba`; 'how many' -> `3`, not `3 albums`."
            ),
        )

    def __call__(
        self,
        question: str,
        attachment_context: str = "",
        images: list[Image.Image] | None = None,
    ) -> str:
        attachment_context = attachment_context.strip()
        if attachment_context:
            question = (
                f"{question}\n\n"
                "Use the following extracted contents from the task's attachment as evidence "
                "when answering:\n"
                f"{attachment_context}"
            )
        result = self.agent.run(question, images=images)
        return _normalize_answer(str(result), question)


class AttachmentUnavailableError(RuntimeError):
    """Raised when the scoring API has no attachment mapped to a task."""


_VISION_SUPPORT_CACHE: dict[str, bool] = {}


def model_supports_vision() -> bool:
    """Ask Ollama whether the configured model accepts images.

    Downloading a video and sampling frames costs minutes; feeding them to a
    text-only model wastes that time and the context they occupy. Assume vision
    works if Ollama cannot be reached, so a non-Ollama backend is unaffected.
    """
    override = os.getenv("LLM_VISION", "auto").strip().lower()
    if override in ("1", "true", "yes"):
        return True
    if override in ("0", "false", "no"):
        return False

    config = resolve_llm_config()
    model_id = config["model_id"]
    if model_id in _VISION_SUPPORT_CACHE:
        return _VISION_SUPPORT_CACHE[model_id]

    # Only Ollama exposes a capability listing; hosted providers do not, and
    # their flagship models are generally multimodal, so assume vision there.
    supported = True
    if config["provider"] == "ollama":
        base_url = config["base_url"].rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        try:
            response = requests.post(
                f"{base_url}/api/show", json={"model": model_id}, timeout=15
            )
            response.raise_for_status()
            supported = "vision" in (response.json().get("capabilities") or [])
        except (requests.exceptions.RequestException, ValueError, KeyError):
            pass  # Cannot tell; assume the model can take images.

    _VISION_SUPPORT_CACHE[model_id] = supported
    return supported


_URL_TRAILING_PUNCTUATION = ".,);:!?'\"»"


def _extract_youtube_video_id(video_url: str) -> str:
    # Models copy URLs straight out of prose, so they usually arrive with the
    # sentence's trailing punctuation attached ("...watch?v=1htKBjuUWec.").
    video_url = video_url.strip().rstrip(_URL_TRAILING_PUNCTUATION)
    parsed_url = urlparse(video_url)
    if parsed_url.hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed_url.path.strip("/").split("/")[0]
    elif parsed_url.hostname and parsed_url.hostname.endswith("youtube.com"):
        video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        if not video_id:
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live"):
                video_id = path_parts[1]
    else:
        video_id = video_url.strip()

    video_id = video_id.strip().rstrip(_URL_TRAILING_PUNCTUATION)
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Could not identify a valid YouTube video ID in the input.")
    return video_id


def extract_youtube_video_frames(video_url: str, max_frames: int = 12) -> list[Image.Image]:
    """Download a YouTube video and sample frames evenly across its duration."""
    import imageio_ffmpeg
    import yt_dlp

    with tempfile.TemporaryDirectory(prefix="agent-video-") as temp_dir:
        output_template = os.path.join(temp_dir, "video.%(ext)s")
        options = {
            "format": "bestvideo[height<=480]/bestvideo[height<=720]/bestvideo/best",
            "outtmpl": output_template,
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "max_filesize": MAX_VIDEO_BYTES,
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                video_info = downloader.extract_info(video_url, download=True)
                video_path = downloader.prepare_filename(video_info)
                if not os.path.exists(video_path):
                    requested_downloads = video_info.get("requested_downloads") or []
                    video_path = (
                        requested_downloads[0].get("filepath", "")
                        if requested_downloads
                        else ""
                    )
                duration = float(video_info.get("duration") or 0)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Could not download the YouTube video: {exc}") from exc

        if not video_path or not os.path.isfile(video_path):
            raise RuntimeError("The YouTube downloader did not produce a video file.")
        if os.path.getsize(video_path) > MAX_VIDEO_BYTES:
            raise ValueError(
                f"The downloaded YouTube video exceeds the {MAX_VIDEO_BYTES}-byte limit."
            )
        if duration <= 0:
            raise ValueError("The YouTube video has no usable duration metadata.")

        frame_pattern = os.path.join(temp_dir, "frame-%02d.jpg")
        interval = max(duration / max(max_frames - 1, 1), 0.5)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps=1/{interval},scale=768:-1",
            "-frames:v", str(max_frames),
            "-y",
            frame_pattern,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not extract YouTube video frames: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out while extracting YouTube video frames.") from exc

        frame_paths = sorted(
            os.path.join(temp_dir, name)
            for name in os.listdir(temp_dir)
            if name.startswith("frame-") and name.endswith(".jpg")
        )
        if not frame_paths:
            raise RuntimeError("No frames could be extracted from the YouTube video.")

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                frames.append(frame.convert("RGB"))
        return frames


def _truncate_to(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Truncated: {len(text) - limit} more characters omitted.]"


def _truncate_extracted_text(text: str) -> str:
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    return (
        f"{text[:MAX_DOCUMENT_CHARS]}\n\n"
        f"[Text truncated: {len(text) - MAX_DOCUMENT_CHARS} additional characters were omitted.]"
    )


def _download_url(url: str) -> tuple[bytes, str]:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError("Only valid HTTP or HTTPS URLs can be downloaded.")

    with requests.get(url, timeout=45, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > MAX_DOCUMENT_BYTES:
                raise ValueError(
                    f"Downloaded content exceeds the {MAX_DOCUMENT_BYTES}-byte limit."
                )
        file_name = os.path.basename(urlparse(response.url).path) or "download"
        return bytes(content), file_name


def _extract_book_text(content: bytes, file_name: str) -> str:
    extension = os.path.splitext(file_name.lower())[1]
    if extension == ".pdf" or content.startswith(b"%PDF"):
        from pypdf import PdfReader

        pages = PdfReader(BytesIO(content)).pages
        text = "\n\n".join(
            f"Page {page_number}: {page.extract_text() or ''}"
            for page_number, page in enumerate(pages, start=1)
        )
    elif extension == ".epub" or content.startswith(b"PK"):
        from bs4 import BeautifulSoup

        with zipfile.ZipFile(BytesIO(content)) as book_archive:
            container = ElementTree.fromstring(
                book_archive.read("META-INF/container.xml")
            )
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None or not rootfile.get("full-path"):
                raise ValueError(f"Could not locate EPUB package metadata in {file_name!r}.")

            package_path = rootfile.attrib["full-path"]
            package = ElementTree.fromstring(book_archive.read(package_path))
            manifest = {
                item.attrib["id"]: item.attrib["href"]
                for item in package.findall(".//{*}manifest/{*}item")
                if item.get("id") and item.get("href")
            }
            chapter_names = [
                posixpath.normpath(
                    posixpath.join(posixpath.dirname(package_path), manifest[itemref.attrib["idref"]])
                )
                for itemref in package.findall(".//{*}spine/{*}itemref")
                if itemref.get("idref") in manifest
            ]
            if not chapter_names:
                raise ValueError(f"No readable chapters were listed in {file_name!r}.")

            chapters = []
            expanded_size = 0
            for member_name in chapter_names:
                if member_name.lower().endswith((".html", ".htm", ".xhtml")):
                    chapter_file = book_archive.getinfo(member_name)
                    expanded_size += chapter_file.file_size
                    if expanded_size > MAX_DOCUMENT_BYTES:
                        raise ValueError(
                            f"EPUB chapters in {file_name!r} exceed the "
                            f"{MAX_DOCUMENT_BYTES}-byte extraction limit."
                        )
                    chapter = BeautifulSoup(
                        book_archive.read(member_name), "html.parser"
                    ).get_text(" ", strip=True)
                    if chapter:
                        chapters.append(chapter)
        text = "\n\n".join(chapters)
    else:
        raise ValueError(
            f"Unsupported book format for {file_name!r}; provide a PDF or EPUB."
        )

    if not text.strip():
        raise ValueError(f"No extractable text was found in {file_name!r}.")
    return _truncate_extracted_text(text)


class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript"
    description = (
        "Retrieves captions/transcript text for a YouTube video URL. "
        "This reads captions only, not visual video frames."
    )
    inputs = {
        "video_url": {
            "type": "string",
            "description": "A YouTube video URL or video ID.",
        }
    }
    output_type = "string"

    def forward(self, video_url: str) -> str:
        from youtube_transcript_api import YouTubeTranscriptApi

        video_id = _extract_youtube_video_id(video_url)

        available = list(YouTubeTranscriptApi().list(video_id))
        if not available:
            raise ValueError(f"No captions are available for YouTube video {video_id}.")

        transcript_info = next(
            (item for item in available if item.language_code.startswith("en")),
            available[0],
        )
        transcript = transcript_info.fetch()
        text = " ".join(segment.text for segment in transcript)
        if not text.strip():
            raise ValueError(f"The YouTube captions for video {video_id} are empty.")
        return (
            f"Transcript language: {transcript_info.language}\n\n"
            f"{_truncate_extracted_text(text)}"
        )


class PoliteVisitWebpageTool(VisitWebpageTool):
    """`visit_webpage` that sends a real User-Agent and a smaller payload.

    smolagents' version calls `requests.get(url)` with no headers, so sites that
    reject the default python-requests agent -- Wikipedia among them -- answer
    403. The smaller default output also keeps hosted rate limits at bay, since
    the whole conversation is resent on every step.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "agents-course-final-assignment/1.0 (educational)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, max_output_length: int = 15000):
        super().__init__(max_output_length=max_output_length)

    def forward(self, url: str) -> str:
        from markdownify import markdownify

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=25)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except requests.exceptions.RequestException as exc:
            return f"Error fetching the webpage: {exc}"

        markdown = markdownify(response.text).strip()
        # markdownify keeps every link target and title, which on a Wikipedia
        # page is mostly navigation chrome and can be over half the payload --
        # tokens that cost rate limit without carrying any answer.
        markdown = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown)
        markdown = re.sub(r"[ \t]+\n", "\n", markdown)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return self._truncate_content(markdown, self.max_output_length)


class WikipediaTool(Tool):
    name = "wikipedia"
    description = (
        "Looks up an English Wikipedia article and returns its source text with TABLES "
        "and section structure intact. Call it with `query` alone to get the article's "
        "opening and a list of its section headings; then call it again with `section` "
        "set to the heading you need (e.g. section='Discography') to read that section's "
        "full contents, including any tables. Prefer this over visit_webpage for "
        "Wikipedia, which blocks automated page fetches."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "An article title or search phrase, e.g. 'Mercedes Sosa'.",
        },
        "section": {
            "type": "string",
            "description": (
                "Optional section heading to return in full, e.g. 'Discography'. "
                "Omit to get the article's opening plus the list of section headings."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    API_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "agents-course-final-assignment/1.0 (educational)"}
    # Deliberately small: hosted free tiers rate-limit on tokens per minute, and
    # smolagents resends the whole history every step.
    MAX_SECTION_CHARS = 6000
    HEADING_PATTERN = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

    def _get(self, params: dict) -> dict:
        params = {"format": "json", "formatversion": "2", **params}
        response = requests.get(
            self.API_URL, params=params, headers=self.HEADERS, timeout=30
        )
        response.raise_for_status()
        return response.json()

    # Templates whose first argument is the content, not formatting. Wikipedia's
    # data tables routinely hide the key column inside these -- the 1928 Olympics
    # athlete counts put every country in {{flagIOC|GER|1928 Summer}} -- so
    # deleting them outright throws away the answer.
    TEMPLATE_KEEP_FIRST_ARG = frozenset(
        {
            "flagioc", "flag", "flagicon", "flagcountry", "flagathlete",
            "flaglist", "flagu", "nts", "sort", "sortname",
        }
    )
    TEMPLATE_KEEP_LAST_ARG = frozenset({"nowrap", "nobr", "lang", "nowraplinks"})
    # Templates whose arguments are all content, e.g. {{sortname|Claus Peter|Flor}}.
    TEMPLATE_JOIN_ARGS = frozenset({"sortname", "plainlist", "ubl", "unbulleted list"})
    # Maintenance templates whose bare name is noise rather than data.
    TEMPLATE_DROP_BARE = frozenset(
        {
            "reflist", "clear", "clr", "authority control", "short description",
            "use dmy dates", "use mdy dates", "commons category", "portal",
            "toc", "tocright", "citation needed", "stub", "expand section",
            "more citations needed", "unreferenced", "orphan",
        }
    )
    TEMPLATE_PATTERN = re.compile(r"\{\{([^{}]*)\}\}")

    @classmethod
    def _replace_template(cls, match: "re.Match[str]") -> str:
        parts = [part.strip() for part in match.group(1).split("|")]
        name = parts[0].strip()
        key = name.lower()
        args = [part for part in parts[1:] if "=" not in part]

        if not args:
            # Country templates carry their value in the NAME and take no
            # arguments -- {{DDR}} is East Germany, {{JAP}} is Japan. Dropping
            # these empties the column the question is about.
            if key in cls.TEMPLATE_DROP_BARE or key.startswith(("cite ", "infobox")):
                return ""
            return name
        if key in cls.TEMPLATE_JOIN_ARGS:
            return " ".join(args)
        if key in cls.TEMPLATE_KEEP_FIRST_ARG:
            return args[0]
        if key in cls.TEMPLATE_KEEP_LAST_ARG:
            return args[-1]
        return ""

    @classmethod
    def _tidy_wikitext(cls, text: str) -> str:
        """Strip the noisiest wikitext markup while leaving tables readable."""
        text = re.sub(r"<ref[^>]*/>", "", text)
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        # Templates use {{...}}; tables use {|...|} and must survive. Resolve
        # innermost-first, repeatedly, so nested templates collapse too.
        for _ in range(5):
            text, replaced = cls.TEMPLATE_PATTERN.subn(cls._replace_template, text)
            if not replaced:
                break
        text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]", r"\1", text)
        text = re.sub(r"'{2,}", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _sections(self, wikitext: str) -> list[tuple[str, str]]:
        """Split wikitext into (heading, body) pairs, with the lead first.

        A section's body runs to the next heading at the same or higher level, so
        asking for 'Discography' also returns its 'Studio albums' subsection.
        """
        matches = list(self.HEADING_PATTERN.finditer(wikitext))
        if not matches:
            return [("(lead)", wikitext)]
        sections = [("(lead)", wikitext[: matches[0].start()])]
        for index, match in enumerate(matches):
            level = len(match.group(1))
            end = len(wikitext)
            for later in matches[index + 1 :]:
                if len(later.group(1)) <= level:
                    end = later.start()
                    break
            sections.append((match.group(2), wikitext[match.end() : end]))
        return sections

    def forward(self, query: str, section: str | None = None) -> str:
        search = self._get(
            {"action": "query", "list": "search", "srsearch": query, "srlimit": 5}
        )
        hits = search.get("query", {}).get("search", [])
        if not hits:
            raise ValueError(f"No English Wikipedia article matched {query!r}.")

        title = hits[0]["title"]
        parsed = self._get({"action": "parse", "page": title, "prop": "wikitext"})
        wikitext = parsed.get("parse", {}).get("wikitext", "")
        if not wikitext.strip():
            raise ValueError(f"The Wikipedia article {title!r} has no readable source.")

        wikitext = self._tidy_wikitext(wikitext)
        sections = self._sections(wikitext)
        header = f"Wikipedia article: {title}"

        if section:
            wanted = section.strip().lower()
            matched = [
                (name, body)
                for name, body in sections
                if wanted == name.lower() or wanted in name.lower()
            ]
            if not matched:
                available = " | ".join(name for name, _ in sections)
                raise ValueError(
                    f"No section matching {section!r} in {title!r}. "
                    f"Available sections: {available}"
                )
            parts = [f"{header} -- section {name!r}:\n{body.strip()}" for name, body in matched]
            return _truncate_to(("\n\n".join(parts)), self.MAX_SECTION_CHARS)

        others = ", ".join(hit["title"] for hit in hits[1:])
        if others:
            header += f"\n(Other matches, if this is the wrong one: {others})"
        lead = sections[0][1].strip()
        headings = " | ".join(name for name, _ in sections[1:]) or "(none)"
        return (
            f"{header}\n\n{_truncate_to(lead, 2000)}\n\n"
            f"SECTIONS AVAILABLE: {headings}\n"
            "Call wikipedia again with section='<one of these>' to read it in full, "
            "including its tables."
        )


class ReadBookTool(Tool):
    name = "read_book"
    description = (
        "Downloads and extracts readable text from an online PDF or EPUB book."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "A direct HTTP or HTTPS URL to a PDF or EPUB book.",
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        content, file_name = _download_url(url)
        return _extract_book_text(content, file_name)


def _transcribe_audio(content: bytes) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Audio transcription requires an HF_TOKEN with access to Hugging Face "
            "Inference Providers. Set HF_TOKEN in the environment and retry."
        )

    from huggingface_hub import InferenceClient

    model_id = os.getenv(
        "AUDIO_TRANSCRIPTION_MODEL", "openai/whisper-large-v3-turbo"
    )
    result = InferenceClient(token=token, timeout=120).automatic_speech_recognition(
        audio=content,
        model=model_id,
    )
    transcript = result if isinstance(result, str) else getattr(result, "text", "")
    if not transcript.strip():
        raise ValueError("The audio transcription service returned no text.")
    return _truncate_extracted_text(transcript)


class TranscribeAudioTool(Tool):
    name = "transcribe_audio"
    description = (
        "Downloads audio from an HTTP or HTTPS URL and transcribes it using "
        "Hugging Face Inference Providers. Requires HF_TOKEN."
    )
    inputs = {
        "audio_url": {
            "type": "string",
            "description": "A direct HTTP or HTTPS URL to an audio file.",
        }
    }
    output_type = "string"

    def forward(self, audio_url: str) -> str:
        content, _ = _download_url(audio_url)
        return _transcribe_audio(content)


def fetch_task_attachment_bytes(task_id: str, file_name: str, api_url: str) -> bytes:
    """Download the task attachment, rejecting unexpectedly large files."""
    file_url = f"{api_url}/files/{task_id}"
    with requests.get(file_url, timeout=30, stream=True) as response:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if response.status_code == 404:
                raise AttachmentUnavailableError(
                    f"The scoring API has no file mapped for task {task_id} "
                    f"({file_name}); it returned HTTP 404."
                ) from exc
            raise RuntimeError(
                f"Could not download attachment {file_name} for task {task_id}: {exc}"
            ) from exc

        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > MAX_DOCUMENT_BYTES:
                raise ValueError(
                    f"Attachment {file_name} exceeds the "
                    f"{MAX_DOCUMENT_BYTES}-byte download limit."
                )
    return bytes(content)


def fetch_excel_attachment_context(content: bytes, file_name: str) -> str:
    """Return Excel worksheets as CSV text for inclusion in the agent context."""
    if not file_name.lower().endswith((".xlsx", ".xlsm")):
        raise ValueError(f"Unsupported Excel attachment format: {file_name}")

    try:
        worksheets = pd.read_excel(
            BytesIO(content),
            sheet_name=None,
            engine="openpyxl",
        )
    except (ValueError, ImportError) as exc:
        raise RuntimeError(
            f"Could not read Excel attachment {file_name}: {exc}"
        ) from exc

    if not worksheets:
        raise ValueError(f"Excel attachment {file_name} contains no worksheets.")

    sections = []
    for sheet_name, dataframe in worksheets.items():
        sections.append(
            f"Sheet: {sheet_name} ({len(dataframe)} rows x {len(dataframe.columns)} columns)\n"
            f"{dataframe.to_csv(index=False)}"
        )

    workbook_text = "\n\n".join(sections)
    if len(workbook_text) > MAX_DOCUMENT_CHARS:
        raise ValueError(
            f"Excel attachment {file_name} is too large to pass to the agent "
            f"({len(workbook_text)} characters; maximum is {MAX_DOCUMENT_CHARS})."
        )
    return workbook_text


# --- Running and submitting are separate steps ---
#
# The batch run performs far worse than running questions one at a time: a
# hosted free tier throttles after a few questions, smolagents sits in retry
# backoff, the agent burns its step budget and returns narration instead of an
# answer. Caching each answer as it lands means a run can be stopped, paced and
# resumed, and the submission can happen later from answers already verified.
ANSWER_CACHE_FILE = os.getenv("ANSWER_CACHE_FILE", "answers_cache.json")
REQUEST_DELAY_SECONDS = float(os.getenv("AGENT_REQUEST_DELAY", "4"))


def _load_answer_cache() -> dict:
    try:
        with open(ANSWER_CACHE_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_answer_cache(cache: dict) -> None:
    with open(ANSWER_CACHE_FILE, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def _is_usable_answer(entry: dict | None) -> bool:
    """A cached entry worth submitting, rather than one worth retrying."""
    answer = ((entry or {}).get("answer") or "").strip()
    return bool(answer) and not answer.startswith("AGENT ERROR")


def _resolve_username(profile, local_username: str) -> str:
    if profile:
        return str(profile.username)
    return local_username.strip()


def fetch_questions(api_url: str = DEFAULT_API_URL) -> list[dict]:
    response = requests.get(f"{api_url}/questions", timeout=30)
    response.raise_for_status()
    questions = response.json()
    if not questions:
        raise ValueError("The scoring API returned an empty question list.")
    return questions


def prepare_task_context(item: dict, api_url: str = DEFAULT_API_URL) -> tuple[str, list]:
    """Collect the evidence text and images available for one task."""
    question_text = item.get("question") or ""
    task_id = item.get("task_id") or ""
    file_name = item.get("file_name") or ""
    attachment_context = ""
    attachment_images: list = []

    video_urls = re.findall(
        r"https?://(?:www\.)?(?:youtube\.com/watch\?[^\s\"<>]+|youtu\.be/[^\s\"<>]+)",
        question_text,
    )
    can_see = model_supports_vision() if (video_urls or file_name) else True
    for video_url in video_urls:
        video_url = video_url.rstrip(".,)")
        if not can_see:
            message = (
                f"Video frames from {video_url} were not sampled: the configured "
                "model has no vision support. Answer from captions via "
                "youtube_transcript or other evidence if possible."
            )
            print(message)
            attachment_context += f"\n\n{message}"
            continue
        try:
            attachment_images.extend(extract_youtube_video_frames(video_url))
        except (RuntimeError, ValueError) as exc:
            message = f"Could not inspect video frames from {video_url}: {exc}"
            print(message)
            attachment_context += f"\n\n{message}"

    extension = os.path.splitext(file_name.lower())[1]
    supported = {
        ".xlsx", ".xlsm", ".pdf", ".epub", ".txt", ".md", ".py",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
        ".mp3", ".mpga", ".wav", ".m4a", ".aac", ".flac", ".ogg",
        ".webm", ".aif", ".aiff",
    }
    if extension not in supported:
        if file_name:
            attachment_context += (
                f"\n\nThe task attachment {file_name} has an unsupported format. "
                "Supported formats are Excel, PDF, EPUB, plain text, Python source, "
                "common image formats, and common audio formats."
            )
        return attachment_context.strip(), attachment_images

    try:
        raw = fetch_task_attachment_bytes(task_id, file_name, api_url)
    except AttachmentUnavailableError as exc:
        message = (
            f"Attachment unavailable: {exc} Answer from other reliable evidence if "
            "possible; say so plainly if the question cannot be answered without it."
        )
        print(message)
        return (attachment_context + f"\n\n{message}").strip(), attachment_images

    if extension in (".xlsx", ".xlsm"):
        attachment_context += (
            f"\n\nExcel workbook {file_name}:\n"
            f"{fetch_excel_attachment_context(raw, file_name)}"
        )
    elif extension in (".pdf", ".epub"):
        attachment_context += (
            f"\n\nExtracted text from attached book/document {file_name}:\n"
            f"{_extract_book_text(raw, file_name)}"
        )
    elif extension in (".txt", ".md"):
        try:
            attachment_context += (
                f"\n\nText from attached file {file_name}:\n"
                f"{_truncate_extracted_text(raw.decode('utf-8'))}"
            )
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Could not decode text attachment {file_name} as UTF-8."
            ) from exc
    elif extension == ".py":
        try:
            source_code = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Could not decode Python attachment {file_name} as UTF-8."
            ) from exc
        attachment_context += (
            f"\n\nSource code from attached Python file {file_name}:\n"
            f"```python\n{_truncate_extracted_text(source_code)}\n```"
        )
    elif extension in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        if not can_see:
            message = (
                f"The image attachment {file_name} was not supplied: the configured "
                "model has no vision support. Answer from the question text and "
                "other evidence if possible."
            )
            print(message)
            attachment_context += f"\n\n{message}"
        else:
            try:
                with Image.open(BytesIO(raw)) as image:
                    attachment_images.append(image.convert("RGB"))
            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(
                    f"Could not read image attachment {file_name}: {exc}"
                ) from exc
    else:
        if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
            message = (
                f"Could not transcribe attached audio {file_name}: "
                "HF_TOKEN is not configured."
            )
            print(message)
            attachment_context += f"\n\n{message}"
        else:
            attachment_context += (
                f"\n\nTranscribed audio from attached file {file_name}:\n"
                f"{_transcribe_audio(raw)}"
            )

    return attachment_context.strip(), attachment_images


def _cache_table(cache: dict, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for item in questions:
        entry = cache.get(item.get("task_id") or "")
        rows.append(
            {
                "Task ID": item.get("task_id"),
                "Question": (item.get("question") or "")[:120],
                "Cached Answer": (entry or {}).get("answer", ""),
                "Status": "cached" if _is_usable_answer(entry)
                else ("error" if entry else "not run"),
            }
        )
    return pd.DataFrame(rows)


def run_agent_on_questions(batch_size: int = 0, only_missing: bool = True):
    """Answer questions and cache each result the moment it lands.

    `batch_size` of 0 means every outstanding question. Answers are written to
    disk after each question, so stopping the run never loses work.
    """
    try:
        questions = fetch_questions()
    except (requests.exceptions.RequestException, ValueError) as exc:
        return f"Could not fetch questions: {exc}", None

    cache = _load_answer_cache()
    outstanding = [
        item
        for item in questions
        if item.get("task_id")
        and item.get("question") is not None
        and not (only_missing and _is_usable_answer(cache.get(item["task_id"])))
    ]
    if not outstanding:
        return (
            f"Nothing to run: {len(questions)} questions, all already cached. "
            "Use 'Submit cached answers' next."
        ), _cache_table(cache, questions)

    todo = outstanding[:batch_size] if batch_size > 0 else outstanding

    try:
        agent = BasicAgent()
    except Exception as exc:
        return f"Could not initialise the agent: {exc}", _cache_table(cache, questions)

    answered = 0
    failed = 0
    for index, item in enumerate(todo, start=1):
        task_id = item["task_id"]
        question_text = item["question"]
        print(f"\n[{index}/{len(todo)}] {task_id}: {question_text[:90]}")
        try:
            context, images = prepare_task_context(item)
            answer = agent(question_text, context, images or None)
            answered += 1
        except Exception as exc:
            answer = f"AGENT ERROR: {type(exc).__name__}: {exc}"
            failed += 1
            print(f"  failed: {answer}")
        cache[task_id] = {
            "question": question_text,
            "answer": answer,
            "answered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_answer_cache(cache)  # Persist per question, not at the end.
        print(f"  -> {answer[:120]!r}")

        # Pace the next request: hosted free tiers throttle per minute, and the
        # backoff that follows is what degrades a long batch run.
        if index < len(todo) and REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    usable = sum(1 for item in questions if _is_usable_answer(cache.get(item.get("task_id") or "")))
    status = (
        f"Ran {len(todo)} question(s): {answered} answered, {failed} failed.\n"
        f"Cached answers: {usable}/{len(questions)}. Cache file: {ANSWER_CACHE_FILE}\n"
    )
    if usable < len(questions):
        status += "Run again to continue with the questions still outstanding."
    else:
        status += "All questions cached -- 'Submit cached answers' is ready."
    print(status)
    return status, _cache_table(cache, questions)


def submit_cached_answers(profile: gr.OAuthProfile | None, local_username: str = ""):
    """Submit whatever is in the cache. Runs no agent, so it is cheap to retry."""
    username = _resolve_username(profile, local_username)
    if not username:
        return "Please log in to Hugging Face or enter your username.", None

    try:
        questions = fetch_questions()
    except (requests.exceptions.RequestException, ValueError) as exc:
        return f"Could not fetch questions: {exc}", None

    cache = _load_answer_cache()
    answers_payload = [
        {"task_id": item["task_id"], "submitted_answer": cache[item["task_id"]]["answer"]}
        for item in questions
        if item.get("task_id") and _is_usable_answer(cache.get(item.get("task_id")))
    ]
    if not answers_payload:
        return (
            "No cached answers to submit. Run the agent first.",
            _cache_table(cache, questions),
        )

    space_id = os.getenv("SPACE_ID")
    submission = {
        "username": username,
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} cached answers for {username}...")
    try:
        response = requests.post(f"{DEFAULT_API_URL}/submit", json=submission, timeout=120)
        response.raise_for_status()
        result = response.json()
        status = (
            f"Submission successful.\n"
            f"User: {result.get('username')}\n"
            f"Score: {result.get('score', 'N/A')}% "
            f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
            f"Submitted {len(answers_payload)}/{len(questions)} questions.\n"
            f"Message: {result.get('message', '')}"
        )
    except requests.exceptions.HTTPError as exc:
        detail = f"status {exc.response.status_code}"
        try:
            detail += f" -- {exc.response.json().get('detail', exc.response.text[:300])}"
        except (requests.exceptions.JSONDecodeError, ValueError):
            detail += f" -- {exc.response.text[:300]}"
        status = f"Submission failed: {detail}"
    except requests.exceptions.RequestException as exc:
        status = f"Submission failed: {exc}"
    print(status)
    return status, _cache_table(cache, questions)


def clear_answer_cache():
    try:
        os.remove(ANSWER_CACHE_FILE)
        status = f"Deleted {ANSWER_CACHE_FILE}."
    except FileNotFoundError:
        status = f"No cache file at {ANSWER_CACHE_FILE}."
    try:
        return status, _cache_table({}, fetch_questions())
    except (requests.exceptions.RequestException, ValueError):
        return status, None


def show_cache():
    cache = _load_answer_cache()
    try:
        questions = fetch_questions()
    except (requests.exceptions.RequestException, ValueError) as exc:
        return f"Could not fetch questions: {exc}", None
    usable = sum(1 for item in questions if _is_usable_answer(cache.get(item.get("task_id") or "")))
    return (
        f"Cached answers: {usable}/{len(questions)} usable. File: {ANSWER_CACHE_FILE}"
    ), _cache_table(cache, questions)


def run_and_submit_all(profile: gr.OAuthProfile | None, local_username: str = ""):
    """Run every outstanding question, then submit -- the original one-shot flow."""
    if not _resolve_username(profile, local_username):
        return "Please log in to Hugging Face or enter your username.", None
    run_status, _ = run_agent_on_questions(batch_size=0)
    submit_status, table = submit_cached_answers(profile, local_username)
    return f"{run_status}\n\n{submit_status}", table


with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        Running and submitting are **separate steps**, because a hosted free tier
        throttles partway through a 20-question batch and the agent then returns
        narration instead of answers.

        1. Enter your Hugging Face username (or log in, on a Space).
        2. **Run a batch** — answer a few questions at a time. Every answer is
           written to the cache file as soon as it lands, so stopping the run
           never loses work. Run it again to continue where it left off.
        3. Check the table. Anything marked `error` is retried by the next run;
           anything `cached` is kept.
        4. **Submit cached answers** when you are satisfied. Submitting runs no
           agent, so it is cheap and can be repeated.

        `AGENT_REQUEST_DELAY` (seconds between questions) and `ANSWER_CACHE_FILE`
        are configurable via the environment.
        """
    )

    running_in_space = bool(os.getenv("SPACE_ID"))
    if running_in_space:
        gr.LoginButton()
        local_username = gr.Textbox(visible=False, value="")
    else:
        local_username = gr.Textbox(
            label="Hugging Face username",
            placeholder="Enter your Hugging Face username",
        )

    with gr.Row():
        batch_size_input = gr.Number(
            label="Questions per run (0 = all outstanding)",
            value=5,
            precision=0,
            minimum=0,
        )
        retry_all_input = gr.Checkbox(
            label="Re-answer questions that are already cached",
            value=False,
        )

    with gr.Row():
        run_button = gr.Button("Run a batch (caches answers)", variant="primary")
        submit_button = gr.Button("Submit cached answers", variant="secondary")
    with gr.Row():
        show_button = gr.Button("Show cache")
        clear_button = gr.Button("Clear cache")

    status_output = gr.Textbox(label="Status", lines=6, interactive=False)
    results_table = gr.DataFrame(label="Questions and cached answers", wrap=True)

    run_button.click(
        fn=lambda size, retry: run_agent_on_questions(
            batch_size=int(size or 0), only_missing=not retry
        ),
        inputs=[batch_size_input, retry_all_input],
        outputs=[status_output, results_table],
    )
    submit_button.click(
        fn=(
            submit_cached_answers
            if running_in_space
            else (lambda username: submit_cached_answers(None, username))
        ),
        inputs=(None if running_in_space else [local_username]),
        outputs=[status_output, results_table],
    )
    show_button.click(fn=show_cache, outputs=[status_output, results_table])
    clear_button.click(fn=clear_answer_cache, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        share=False,
    )