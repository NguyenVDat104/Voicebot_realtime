import os
import ssl
import uuid
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

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


from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from audio_processing.vad import SileroVADDetector
from audio_processing.stt import FasterWhisperSTT
from audio_processing.tts import EdgeTTSStreamer, split_into_sentences
from services.n8n_client import N8NClient

# Load environment variables
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("voicebot.main")

# Global engine instances
stt_engine: Optional[FasterWhisperSTT] = None
tts_engine: Optional[EdgeTTSStreamer] = None
n8n_client: Optional[N8NClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and shared clients during startup."""
    global stt_engine, tts_engine, n8n_client
    logger.info("Initializing Voicebot Audio Gateway components...")

    # Faster Whisper STT
    whisper_model = os.getenv("WHISPER_MODEL_SIZE", "small")
    whisper_device = os.getenv("WHISPER_DEVICE", "auto")
    whisper_compute = os.getenv("WHISPER_COMPUTE_TYPE", "default")
    stt_engine = FasterWhisperSTT(
        model_size=whisper_model,
        device=whisper_device,
        compute_type=whisper_compute,
        language="vi",
        beam_size=3,
    )

    # Edge TTS
    tts_voice = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
    tts_rate = os.getenv("TTS_RATE", "+0%")
    tts_pitch = os.getenv("TTS_PITCH", "+0Hz")
    tts_engine = EdgeTTSStreamer(voice=tts_voice, rate=tts_rate, pitch=tts_pitch)

    # n8n Client
    n8n_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/voicebot")
    n8n_timeout = float(os.getenv("N8N_TIMEOUT_SECONDS", "10.0"))
    n8n_client = N8NClient(webhook_url=n8n_url, timeout=n8n_timeout)

    logger.info("Voicebot Audio Gateway initialized and ready.")
    yield
    logger.info("Shutting down Voicebot Audio Gateway...")

app = FastAPI(title="Real-Time Vietnamese Voicebot", lifespan=lifespan)

# Enable CORS for cross-origin integration (web widgets, mobile apps, remote dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.get("/")
async def get_index():
    """Serves the frontend application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Voicebot API is running. index.html not yet created in static/"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "stt_ready": stt_engine is not None and stt_engine.model is not None,
        "tts_voice": tts_engine.voice if tts_engine else None,
        "n8n_webhook": n8n_client.webhook_url if n8n_client else None,
    }


# ==============================================================================
# AUTHENTICATION & USER MANAGEMENT API
# ==============================================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str

USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "123",  # Mật khẩu demo mặc định
        "name": "Nguyễn Văn Đạt",
        "email": "dat.nguyen@voicebot.ai",
        "role": "Kỹ sư Trí tuệ Nhân tạo (Admin)",
        "avatar": "👨‍💻",
    },
    "guest": {
        "username": "guest",
        "password": "123",
        "name": "Khách thử nghiệm",
        "email": "guest@voicebot.ai",
        "role": "Thành viên trải nghiệm",
        "avatar": "👤",
    }
}
ACTIVE_SESSIONS = {}

@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    """Xác thực đăng nhập và cấp token phiên."""
    uname = req.username.strip().lower()
    user = USERS_DB.get(uname)
    if not user or user["password"] != req.password.strip():
        # Cho phép bất kỳ username nào với password bất kỳ để tiện demo linh hoạt nếu chưa tồn tại
        if uname and len(uname) >= 3 and len(req.password.strip()) >= 3:
            user = {
                "username": uname,
                "password": req.password.strip(),
                "name": req.username.capitalize(),
                "email": f"{uname}@voicebot.ai",
                "role": "Thành viên Voicebot",
                "avatar": "🧑‍🚀",
            }
            USERS_DB[uname] = user
        else:
            raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác (Tối thiểu 3 ký tự).")

    token = f"token_{uuid.uuid4().hex}"
    user_info = {
        "username": user["username"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "avatar": user["avatar"],
    }
    ACTIVE_SESSIONS[token] = user_info
    return {
        "success": True,
        "token": token,
        "user": user_info,
        "message": f"Chào mừng {user['name']} quay trở lại!",
    }

@app.post("/api/auth/register")
async def auth_register(req: RegisterRequest):
    """Đăng ký tài khoản người dùng mới."""
    uname = req.username.strip().lower()
    if not uname or len(uname) < 3:
        raise HTTPException(status_code=400, detail="Tên đăng nhập phải có ít nhất 3 ký tự.")
    if uname in USERS_DB:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại trên hệ thống.")

    user = {
        "username": uname,
        "password": req.password.strip(),
        "name": req.name.strip() or uname.capitalize(),
        "email": f"{uname}@voicebot.ai",
        "role": "Thành viên mới",
        "avatar": "🌟",
    }
    USERS_DB[uname] = user

    token = f"token_{uuid.uuid4().hex}"
    user_info = {
        "username": user["username"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "avatar": user["avatar"],
    }
    ACTIVE_SESSIONS[token] = user_info
    return {
        "success": True,
        "token": token,
        "user": user_info,
        "message": f"Đăng ký tài khoản thành công!",
    }

@app.get("/api/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    """Lấy thông tin người dùng hiện tại từ token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập hoặc thiếu token.")
    token = authorization.replace("Bearer ", "").strip()
    user = ACTIVE_SESSIONS.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập đã hết hạn.")
    return {"success": True, "user": user}

@app.post("/api/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    """Hủy token phiên đăng xuất."""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        ACTIVE_SESSIONS.pop(token, None)
    return {"success": True, "message": "Đã đăng xuất thành công."}

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Full-Duplex Real-Time WebSocket endpoint:
    - Concurrent non-blocking receive loop.
    - Continuous VAD processing with dynamic thresholding.
    - Instant Barge-in: user speech cancels ongoing TTS/LLM pipeline and stops client audio.
    - Continuous hands-free conversational loop.
    """
    await websocket.accept()

    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket client connected. Assigned session_id: {session_id}")

    # Read VAD configuration from env or use defaults
    vad_threshold = float(os.getenv("VAD_CONFIDENCE_THRESHOLD", "0.5"))
    vad_silence_ms = int(os.getenv("VAD_SILENCE_DURATION_MS", "700"))
    vad_min_speech_ms = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "250"))

    vad_detector = SileroVADDetector(
        sample_rate=16000,
        threshold=vad_threshold,
        silence_duration_ms=vad_silence_ms,
        min_speech_duration_ms=vad_min_speech_ms,
    )

    # State tracking
    current_pipeline_task: Optional[asyncio.Task] = None
    is_bot_speaking = False
    is_pipeline_running = False
    ws_lock = asyncio.Lock()
    session_voice = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
    session_rate = os.getenv("TTS_RATE", "+0%")

    async def safe_send_json(data: dict):
        """Thread-safe and coroutine-safe JSON sender to prevent ASGI protocol race conditions."""
        try:
            async with ws_lock:
                await websocket.send_json(data)
        except Exception as send_err:
            logger.debug(f"safe_send_json exception (session {session_id}): {send_err}")

    async def safe_send_bytes(data: bytes):
        """Thread-safe and coroutine-safe byte chunk sender."""
        try:
            async with ws_lock:
                await websocket.send_bytes(data)
        except Exception as send_err:
            logger.debug(f"safe_send_bytes exception (session {session_id}): {send_err}")

    async def cancel_current_pipeline(reason: str = "Interrupted"):
        """Instantly cancels any running STT / LLM / TTS task and informs client."""
        nonlocal current_pipeline_task, is_bot_speaking, is_pipeline_running
        if current_pipeline_task and not current_pipeline_task.done():
            logger.info(f"Cancelling active pipeline task for session {session_id} ({reason})")
            current_pipeline_task.cancel()
            try:
                await current_pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
            current_pipeline_task = None

        is_bot_speaking = False
        is_pipeline_running = False
        vad_detector.set_bot_state(is_speaking=False, is_processing=False)

        # Notify client to immediately halt audio playback
        await safe_send_json({
            "type": "interrupt",
            "reason": reason,
        })

    async def run_pipeline(complete_utterance: np.ndarray):
        """Asynchronous pipeline: STT -> LLM (n8n) -> Sentence-level Streaming TTS."""
        nonlocal is_bot_speaking, is_pipeline_running
        t_total_start = time.perf_counter()
        try:
            is_pipeline_running = True
            is_bot_speaking = False
            vad_detector.set_bot_state(is_speaking=False, is_processing=True)
            logger.info(f"Processing utterance of {len(complete_utterance)/16000:.2f}s...")

            # 1. Inform client: Processing STT
            await safe_send_json({
                "type": "status",
                "status": "processing",
                "detail": "Đang nhận diện giọng nói...",
            })

            # 2. Transcribe audio via STT engine
            t_stt_start = time.perf_counter()
            transcription = ""
            if stt_engine:
                transcription = await stt_engine.transcribe(complete_utterance)
            stt_duration_ms = (time.perf_counter() - t_stt_start) * 1000

            if not transcription.strip():
                logger.info("Empty transcription, returning to listening.")
                is_pipeline_running = False
                vad_detector.set_bot_state(is_speaking=False, is_processing=False)
                if not vad_detector.is_speaking:
                    vad_detector.reset()
                await safe_send_json({"type": "status", "status": "listening", "detail": "Đang lắng nghe..."})
                return

            # 3. Send recognized transcript
            await safe_send_json({
                "type": "user_transcript",
                "text": transcription,
            })

            # 4. Inform client: Calling n8n / LLM
            await safe_send_json({
                "type": "status",
                "status": "thinking",
                "detail": "Đang suy nghĩ câu trả lời...",
            })

            # 5. Call n8n webhook
            t_llm_start = time.perf_counter()
            ai_reply = ""
            if n8n_client:
                ai_reply = await n8n_client.send_message(session_id, transcription)
            else:
                ai_reply = "Xin chào, tôi đã nghe rõ câu hỏi của bạn."
            llm_duration_ms = (time.perf_counter() - t_llm_start) * 1000

            # 6. Send bot response text
            await safe_send_json({
                "type": "bot_response",
                "text": ai_reply,
            })

            # 7. Start Sentence-level TTS streaming
            is_bot_speaking = True
            is_pipeline_running = False  # Processing phase done, playback phase begins
            vad_detector.set_bot_state(is_speaking=True, is_processing=False)

            await safe_send_json({
                "type": "status",
                "status": "speaking",
                "detail": "Voicebot đang nói...",
            })
            await safe_send_json({"type": "tts_stream_start"})

            # Split text into sentences for rapid streaming (<400ms to first sentence audio)
            sentences = split_into_sentences(ai_reply)
            total_sentences = len(sentences)
            logger.info(f"Pipelining {total_sentences} sentences for session {session_id} TTS streaming (voice={session_voice}, rate={session_rate})...")

            tts_first_chunk_recorded = False
            tts_ttfa_ms = 0.0

            if tts_engine and total_sentences > 0:
                t_tts_start = time.perf_counter()
                for idx, sentence in enumerate(sentences):
                    # If user barged in, abort remaining sentences immediately
                    if not is_bot_speaking:
                        logger.info(f"Sentence generation aborted at {idx}/{total_sentences} due to interruption.")
                        break

                    logger.debug(f"Streaming sentence {idx+1}/{total_sentences}: '{sentence}'")
                    await safe_send_json({
                        "type": "tts_sentence_start",
                        "sentence_index": idx,
                        "total_sentences": total_sentences,
                        "text": sentence,
                    })

                    chunk_count = 0
                    async for chunk_bytes in tts_engine.stream_audio_chunks(sentence, voice=session_voice, rate=session_rate):
                        if not is_bot_speaking:
                            break
                        if not tts_first_chunk_recorded:
                            tts_first_chunk_recorded = True
                            tts_ttfa_ms = (time.perf_counter() - t_tts_start) * 1000
                            total_turnaround_ms = (time.perf_counter() - t_total_start) * 1000
                            # Realtime Latency metrics payload sent to client!
                            await safe_send_json({
                                "type": "latency_metrics",
                                "stt_ms": round(stt_duration_ms, 1),
                                "llm_ms": round(llm_duration_ms, 1),
                                "tts_ms": round(tts_ttfa_ms, 1),
                                "total_ms": round(total_turnaround_ms, 1),
                                "voice": session_voice,
                                "rate": session_rate,
                            })

                        chunk_count += 1
                        await safe_send_bytes(chunk_bytes)

                    await safe_send_json({
                        "type": "tts_sentence_end",
                        "sentence_index": idx,
                    })

            await safe_send_json({"type": "tts_stream_end"})
            await safe_send_json({"type": "tts_all_done"})

        except asyncio.CancelledError:
            logger.info("Pipeline task was cancelled (Barge-in).")
            raise
        except Exception as err:
            logger.error(f"Error in pipeline task: {err}", exc_info=True)
            is_bot_speaking = False
            is_pipeline_running = False
            vad_detector.set_bot_state(is_speaking=False, is_processing=False)
            await safe_send_json({
                "type": "status",
                "status": "listening",
                "detail": "Có lỗi xử lý, đang lắng nghe tiếp...",
            })

    async def run_text_pipeline(prompt_text: str):
        """Pipeline triggered directly by keyboard text input (bypasses STT)."""
        nonlocal is_bot_speaking, is_pipeline_running
        t_total_start = time.perf_counter()
        try:
            is_pipeline_running = True
            is_bot_speaking = False
            vad_detector.set_bot_state(is_speaking=False, is_processing=True)

            # 1. Inform client: Calling n8n / LLM
            await safe_send_json({
                "type": "status",
                "status": "thinking",
                "detail": "Đang suy nghĩ câu trả lời...",
            })

            # 2. Call n8n webhook
            t_llm_start = time.perf_counter()
            ai_reply = ""
            if n8n_client:
                ai_reply = await n8n_client.send_message(session_id, prompt_text)
            else:
                ai_reply = "Xin chào, tôi đã nhận được câu hỏi dạng văn bản của bạn."
            llm_duration_ms = (time.perf_counter() - t_llm_start) * 1000

            # 3. Send bot response text
            await safe_send_json({
                "type": "bot_response",
                "text": ai_reply,
            })

            # 4. Start Sentence-level TTS streaming
            is_bot_speaking = True
            is_pipeline_running = False
            vad_detector.set_bot_state(is_speaking=True, is_processing=False)

            await safe_send_json({
                "type": "status",
                "status": "speaking",
                "detail": "Voicebot đang nói...",
            })
            await safe_send_json({"type": "tts_stream_start"})

            sentences = split_into_sentences(ai_reply)
            total_sentences = len(sentences)
            logger.info(f"Pipelining {total_sentences} text-initiated sentences for session {session_id} (voice={session_voice}, rate={session_rate})...")

            tts_first_chunk_recorded = False
            tts_ttfa_ms = 0.0

            if tts_engine and total_sentences > 0:
                t_tts_start = time.perf_counter()
                for idx, sentence in enumerate(sentences):
                    if not is_bot_speaking:
                        logger.info(f"Sentence generation aborted at {idx}/{total_sentences} due to interruption.")
                        break

                    await safe_send_json({
                        "type": "tts_sentence_start",
                        "sentence_index": idx,
                        "total_sentences": total_sentences,
                        "text": sentence,
                    })

                    async for chunk_bytes in tts_engine.stream_audio_chunks(sentence, voice=session_voice, rate=session_rate):
                        if not is_bot_speaking:
                            break
                        if not tts_first_chunk_recorded:
                            tts_first_chunk_recorded = True
                            tts_ttfa_ms = (time.perf_counter() - t_tts_start) * 1000
                            total_turnaround_ms = (time.perf_counter() - t_total_start) * 1000
                            await safe_send_json({
                                "type": "latency_metrics",
                                "stt_ms": 0.0,
                                "llm_ms": round(llm_duration_ms, 1),
                                "tts_ms": round(tts_ttfa_ms, 1),
                                "total_ms": round(total_turnaround_ms, 1),
                                "voice": session_voice,
                                "rate": session_rate,
                            })

                        await safe_send_bytes(chunk_bytes)

                    await safe_send_json({
                        "type": "tts_sentence_end",
                        "sentence_index": idx,
                    })

            await safe_send_json({"type": "tts_stream_end"})
            await safe_send_json({"type": "tts_all_done"})

        except asyncio.CancelledError:
            logger.info("Text pipeline task was cancelled.")
            raise
        except Exception as err:
            logger.error(f"Error in text pipeline task: {err}", exc_info=True)
            is_bot_speaking = False
            is_pipeline_running = False
            vad_detector.set_bot_state(is_speaking=False, is_processing=False)
            await safe_send_json({
                "type": "status",
                "status": "listening",
                "detail": "Có lỗi xử lý, đang lắng nghe tiếp...",
            })

    # Send initial session established message
    await safe_send_json({
        "type": "session_init",
        "session_id": session_id,
        "voice": session_voice,
        "rate": session_rate,
        "message": "Connected to Vietnamese Voicebot Gateway (Full-Duplex)",
    })

    try:
        while True:
            # Idle timeout: close session if no message/audio received within 120s (cleans up zombie connections)
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.info(f"WebSocket session {session_id} timed out after 120s of inactivity. Closing.")
                await safe_send_json({
                    "type": "timeout",
                    "message": "Kết nối đã tự động đóng sau 120 giây không hoạt động.",
                })
                break

            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client disconnected: {session_id}")
                break

            # Handle text control events
            if "text" in message and message["text"]:
                try:
                    import json
                    event_data = json.loads(message["text"])
                    event_type = event_data.get("type")

                    if event_type == "set_session":
                        custom_sid = event_data.get("session_id")
                        if custom_sid:
                            session_id = custom_sid
                            logger.info(f"Updated session_id to: {session_id}")

                    elif event_type == "user_text":
                        text_val = event_data.get("text", "").strip()
                        if text_val:
                            logger.info(f"Received manual user text for session {session_id}: '{text_val}'")
                            if (current_pipeline_task and not current_pipeline_task.done()) or is_bot_speaking:
                                await cancel_current_pipeline("New text question submitted")
                            current_pipeline_task = asyncio.create_task(run_text_pipeline(text_val))

                    elif event_type == "force_utterance_complete":
                        logger.info(f"Push-to-Talk released for session {session_id}, checking speech buffer...")
                        if len(vad_detector.speech_buffer) > 0:
                            complete_utterance = np.concatenate(vad_detector.speech_buffer)
                            vad_detector.reset()
                            if len(complete_utterance) >= vad_detector.min_speech_samples:
                                if (current_pipeline_task and not current_pipeline_task.done()) or is_bot_speaking:
                                    await cancel_current_pipeline("Push-to-talk new utterance")
                                current_pipeline_task = asyncio.create_task(run_pipeline(complete_utterance))

                    elif event_type == "update_settings":
                        new_silence_ms = event_data.get("silence_duration_ms")
                        new_threshold = event_data.get("threshold")
                        new_voice = event_data.get("voice")
                        new_rate = event_data.get("rate")
                        if new_silence_ms is not None:
                            vad_detector.min_silence_samples = int((float(new_silence_ms) / 1000.0) * vad_detector.sample_rate)
                            logger.info(f"Updated VAD silence duration to {new_silence_ms}ms")
                        if new_threshold is not None:
                            vad_detector.threshold = float(new_threshold)
                            logger.info(f"Updated VAD threshold to {new_threshold}")
                        if new_voice:
                            session_voice = str(new_voice)
                            logger.info(f"Updated TTS voice to: {session_voice}")
                        if new_rate:
                            session_rate = str(new_rate)
                            logger.info(f"Updated TTS rate to: {session_rate}")

                    elif event_type == "playback_ended":
                        # Client finished playing TTS audio
                        is_bot_speaking = False
                        is_pipeline_running = False
                        vad_detector.set_bot_state(is_speaking=False, is_processing=False)
                        # Keep user speech if they started speaking right before playback_ended
                        if not vad_detector.is_speaking:
                            vad_detector.reset()
                        await safe_send_json({
                            "type": "status",
                            "status": "listening",
                            "detail": "Đang lắng nghe... Hãy nói tiếp",
                        })

                    elif event_type == "client_interrupt":
                        # Explicit interruption requested by client
                        logger.info("Client requested explicit interruption.")
                        await cancel_current_pipeline("Client explicit interrupt")
                        if not vad_detector.is_speaking:
                            vad_detector.reset()

                    elif event_type == "ping":
                        await safe_send_json({"type": "pong"})

                except Exception as ex:
                    logger.warning(f"Error parsing JSON text message: {ex}")

            # Handle binary messages (PCM 16-bit audio from mic)
            elif "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]

                # Convert PCM 16-bit bytes (Int16) to 16kHz Float32 [-1.0, 1.0]
                try:
                    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    audio_float32 = pcm_int16.astype(np.float32) / 32768.0
                except Exception as conv_err:
                    logger.warning(f"Error converting audio bytes: {conv_err}")
                    continue

                # Run through VAD detector
                is_speaking, complete_utterance, barge_in = vad_detector.process_chunk(audio_float32)

                # BARGE-IN DETECTED: User spoke while bot was speaking or processing
                if barge_in and (is_bot_speaking or is_pipeline_running):
                    logger.info(f"Barge-in triggered for session {session_id}! Cancelling bot output...")
                    await cancel_current_pipeline("Barge-in voice detected")
                    await safe_send_json({
                        "type": "status",
                        "status": "listening",
                        "detail": "⚡ Đã ngắt lời bot! Đang nghe bạn nói...",
                    })

                # Real-time speech activity for client visualizer
                if is_speaking:
                    await safe_send_json({"type": "vad_activity", "is_speaking": True})

                # Complete utterance detected!
                if complete_utterance is not None:
                    # If any previous task is running OR if bot was still speaking/playing on client:
                    # ALWAYS cancel and silence previous output to guarantee voices never overlap!
                    if (current_pipeline_task and not current_pipeline_task.done()) or is_bot_speaking:
                        logger.info(f"Silencing previous playback for session {session_id} because new utterance arrived.")
                        await cancel_current_pipeline("New utterance started while bot was speaking")

                    # Dispatch pipeline as an asynchronous background task so receive loop continues immediately!
                    current_pipeline_task = asyncio.create_task(run_pipeline(complete_utterance))

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}", exc_info=True)
    finally:
        if current_pipeline_task and not current_pipeline_task.done():
            current_pipeline_task.cancel()
        logger.info(f"Cleaned up session {session_id}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
