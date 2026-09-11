import asyncio
import logging
import re
import ssl
from typing import AsyncGenerator, List

# Fix for Windows certificate store [ASN1: NOT_ENOUGH_DATA] bug
try:
    import certifi
    _orig_load_default_certs = ssl.SSLContext.load_default_certs
    def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
        try:
            _orig_load_default_certs(self, purpose)
        except Exception:
            self.load_verify_locations(cafile=certifi.where())
    ssl.SSLContext.load_default_certs = _safe_load_default_certs
except Exception:
    pass

import edge_tts

logger = logging.getLogger("voicebot.tts")

# Common abbreviations in Vietnamese to avoid splitting on dots
ABBREVIATIONS = {
    "TS.": "Tiến sĩ ",
    "ThS.": "Thạc sĩ ",
    "PGS.": "Phó giáo sư ",
    "GS.": "Giáo sư ",
    "TP.": "Thành phố ",
    "tp.": "thành phố ",
    "v.v.": "vân vân",
    "v.v": "vân vân",
    "vs.": "với ",
    "approx.": "khoảng ",
}


def sanitize_text_for_tts(text: str) -> str:
    """
    Cleans raw LLM text by removing Markdown formatting, URLs, emojis,
    and unnecessary symbols so that Edge-TTS sounds natural and fluent.
    """
    if not text:
        return ""

    cleaned = text

    # 1. Replace known abbreviations
    for abbr, full in ABBREVIATIONS.items():
        cleaned = cleaned.replace(abbr, full)

    # 2. Remove Markdown links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)

    # 3. Remove raw URLs (http://... or https://...)
    cleaned = re.sub(r'https?://\S+', '', cleaned)

    # 4. Remove Markdown formatting (*, **, _, __, #, `, ~)
    cleaned = re.sub(r'(\*\*|__)(.*?)\1', r'\2', cleaned)
    cleaned = re.sub(r'(\*|_)(.*?)\1', r'\2', cleaned)
    cleaned = re.sub(r'~~(.*?)~~', r'\1', cleaned)
    cleaned = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', cleaned)
    cleaned = re.sub(r'#+\s*', '', cleaned)

    # 5. Remove Markdown list bullets and blockquotes at line starts
    cleaned = re.sub(r'^\s*[-*+]\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*>\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)

    # 6. Remove Unicode Emojis and miscellaneous symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "\u2600-\u26FF"          # miscellaneous symbols
        "\u2700-\u27BF"          # dingbats
        "\u2300-\u23FF"          # misc technical
        "]+",
        flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub('', cleaned)

    # 7. Remove remaining standalone special characters (*, #, @, $, %, ^, &, _, |, <, >)
    cleaned = re.sub(r'[*#@$%^&_|~<>{}[\]]', ' ', cleaned)

    # 8. Collapse repeated punctuation (e.g. "??" -> "?", "!!" -> "!")
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'\?{2,}', '?', cleaned)
    cleaned = re.sub(r'!{2,}', '!', cleaned)

    # 9. Clean up whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into natural conversational sentences for rapid streaming TTS.
    Preserves numbers (e.g. 3.14) and joins tiny fragments.
    """
    cleaned = sanitize_text_for_tts(text)
    if not cleaned:
        return []

    # Regex sentence boundary:
    # Match punctuation (. ? ! \n ;) followed by space or end of string,
    # avoiding splits within numbers like 3.14 (negative lookbehind/lookahead for digits).
    sentence_regex = re.compile(r'(?<!\d)[.?!;]+(?!\d)\s*|\n+')
    raw_segments = sentence_regex.split(cleaned)

    # Extract delimiter marks to attach back if desired
    delimiters = sentence_regex.findall(cleaned)

    sentences: List[str] = []
    for i, seg in enumerate(raw_segments):
        s = seg.strip()
        if not s:
            continue
        # Attach back ending punctuation if exists
        if i < len(delimiters):
            delim = delimiters[i].strip()
            if delim and not s.endswith(('.', '?', '!', ';')):
                s += delim
        sentences.append(s)

    if not sentences:
        return [cleaned]

    # Post-processing: Merge very short fragments (< 3 words) into adjacent sentences
    # to avoid awkward choppy cadence in Edge-TTS.
    merged_sentences: List[str] = []
    buffer = ""

    for s in sentences:
        words = s.split()
        if len(words) < 3 and buffer:
            # Append to buffer
            buffer = f"{buffer} {s}"
        else:
            if buffer:
                merged_sentences.append(buffer.strip())
            buffer = s

    if buffer:
        merged_sentences.append(buffer.strip())

    return merged_sentences if merged_sentences else [cleaned]


class EdgeTTSStreamer:
    """
    Text-to-Speech engine using edge-tts.
    Provides streaming generation of audio chunks (MP3 format) to minimize time-to-first-byte latency.
    Supports sentence-level pipelining for sub-400ms conversational response.
    """

    def __init__(
        self,
        voice: str = "vi-VN-HoaiMyNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def stream_audio_chunks(
        self,
        text: str,
        voice: str = None,
        rate: str = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Streams audio bytes chunk-by-chunk directly using edge_tts.Communicate.stream().
        Yields raw audio bytes (MP3) as soon as each chunk is received from the server.
        """
        if not text or not text.strip():
            return

        selected_voice = voice or self.voice
        selected_rate = rate or self.rate
        sanitized = sanitize_text_for_tts(text)
        if not sanitized:
            return

        try:
            communicate = edge_tts.Communicate(
                text=sanitized,
                voice=selected_voice,
                rate=selected_rate,
                pitch=self.pitch,
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"Error streaming TTS from edge-tts: {e}", exc_info=True)

