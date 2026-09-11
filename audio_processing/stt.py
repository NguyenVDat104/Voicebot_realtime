import asyncio
import io
import logging
import os
from typing import Optional
import httpx
import numpy as np
import wave

logger = logging.getLogger("voicebot.stt")

# Common hallucination phrases from YouTube subtitles
HALLUCINATION_PHRASES = [
    "subscribe",
    "la la school",
    "ghiền mì gõ",
    "đăng ký kênh",
    "hẹn gặp lại",
    "phần chi tiền thiệt",
    "cảm ơn các bạn đã theo dõi",
    "hãy like và share",
]

class FasterWhisperSTT:
    """
    State-of-the-Art Vietnamese Speech-to-Text engine.
    - Primary Cloud: Groq API (whisper-large-v3) if GROQ_API_KEY is configured (ultra-fast ~200ms).
    - Primary Local: VinAI PhoWhisper (mad1999/pho-whisper-small-ct2) fine-tuned specifically for Vietnamese.
    - Anti-hallucination VAD filtering and peak amplitude normalization.
    """

    def __init__(
        self,
        model_size: str = "mad1999/pho-whisper-small-ct2",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "vi",
        beam_size: int = 5,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.model = None

        # Check Groq API configuration
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if self.groq_api_key:
            logger.info("Groq API Key detected! Groq Cloud STT (whisper-large-v3) enabled.")

        self._init_model()

    def _init_model(self):
        """Initializes the local faster-whisper / PhoWhisper model."""
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading local STT model '{self.model_size}' on device '{self.device}'...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Local STT model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load local model ({e}), fallback enabled.")
            self.model = None

    async def _transcribe_groq(self, audio_float32: np.ndarray) -> Optional[str]:
        """Sends audio to Groq's ultra-fast Whisper Large V3 API (~200ms)."""
        if not self.groq_api_key:
            return None

        try:
            # Convert float32 16kHz audio to WAV in-memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_float32 * 32767).astype(np.int16).tobytes())
            wav_bytes = wav_io.getvalue()

            async with httpx.AsyncClient(timeout=10.0) as client:
                files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
                data = {
                    "model": "whisper-large-v3",
                    "language": "vi",
                    "temperature": "0.0",
                    "response_format": "json",
                }
                headers = {"Authorization": f"Bearer {self.groq_api_key}"}

                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                )

                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("text", "").strip()
                    logger.info(f"Groq STT Result: '{transcript}'")
                    return transcript
                else:
                    logger.warning(f"Groq STT API returned {response.status_code}: {response.text}")
                    return None
        except Exception as e:
            logger.warning(f"Error calling Groq STT API ({e}), falling back to local model.")
            return None

    def _transcribe_local_blocking(self, audio: np.ndarray) -> str:
        """Synchronous local PhoWhisper inference with VAD filtering."""
        if self.model is None:
            return "Xin chào, tôi là trợ lý ảo tiếng Việt."

        try:
            # vad_filter=True removes silence/noise chunks preventing subtitle hallucinations
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt="Đây là câu hỏi giao tiếp tiếng Việt thông thường của người dùng với trợ lý ảo.",
                repetition_penalty=1.2,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )

            transcription = "".join([segment.text for segment in segments]).strip()
            logger.info(f"Local STT Result (lang={info.language}): '{transcription}'")
            return transcription
        except Exception as e:
            logger.error(f"Error during local STT inference: {e}", exc_info=True)
            return ""

    async def transcribe(self, audio_float32: np.ndarray) -> str:
        """
        Public transcription method:
        1. Normalizes audio & filters background noise.
        2. Tries Groq Cloud API if key is present.
        3. Falls back to local PhoWhisper.
        4. Filters out known YouTube subtitle hallucinations.
        """
        if len(audio_float32) == 0:
            return ""

        # 1. Clean DC offset and check audio volume
        audio = audio_float32.astype(np.float32)
        audio = audio - np.mean(audio)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))

        # Discard pure noise/hum/whisper below speech threshold
        if rms < 0.008 or peak < 0.025:
            logger.info(f"Audio energy too low for speech (rms={rms:.4f}, peak={peak:.4f}), ignoring.")
            return ""

        # Peak normalization to 0.95
        audio = (audio / peak) * 0.95

        # 2. Try Groq API if configured
        if self.groq_api_key:
            groq_res = await self._transcribe_groq(audio)
            if groq_res is not None:
                transcription = groq_res
            else:
                transcription = await asyncio.to_thread(self._transcribe_local_blocking, audio)
        else:
            # 3. Local PhoWhisper
            transcription = await asyncio.to_thread(self._transcribe_local_blocking, audio)

        # 4. Anti-hallucination filter
        lower = transcription.lower()
        if any(h in lower for h in HALLUCINATION_PHRASES):
            logger.warning(f"Discarded hallucination text: '{transcription}'")
            return ""

        return transcription
