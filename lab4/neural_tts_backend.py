"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4
Mô-đun Pretrained Neural TTS Offline Wrapper (Yêu cầu 4.6 & 5.6)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Đặc tả:
  API chuẩn: y_nn, sr, extra = synthesize(text)
  - Trả về waveform (float32, 16kHz)
  - extra chứa: thời gian inference (latency_ms), mel-spectrogram, rt_factor (Real-Time Factor)
  - Hỗ trợ local cache để chạy hoàn toàn offline không phụ thuộc internet trong phòng Lab.
===============================================================================
"""

import os
import sys
import time
import hashlib
import asyncio
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "lab4_data" / "neural_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def _run_edge_tts_async(text, out_mp3, voice="vi-VN-HoaiMyNeural"):
    import edge_tts
    comm = edge_tts.Communicate(text=text, voice=voice)
    await comm.save(out_mp3)


def synthesize(text, sr=TARGET_SR, voice="vi-VN-HoaiMyNeural"):
    """
    API chính theo yêu cầu đề bài (Mục 5.6):
    y_nn, sr, extra = synthesize("Mời khách hàng số 23 đến quầy 5")
    """
    t0 = time.perf_counter()
    text_hash = hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:12]
    cached_wav = CACHE_DIR / f"neural_{text_hash}.wav"

    # 1. Kiểm tra cache offline trước
    if cached_wav.exists():
        y, file_sr = sf.read(str(cached_wav), dtype="float32")
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
    else:
        # Nếu chưa cache, sinh bằng edge-tts và lưu vào cache offline
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_f:
            tmp_mp3 = tmp_f.name

        try:
            asyncio.run(_run_edge_tts_async(text, tmp_mp3, voice=voice))
            y, _ = librosa.load(tmp_mp3, sr=sr, mono=True)
            # Chuẩn hóa âm lượng nhẹ
            y = y / (np.max(np.abs(y)) + 1e-8) * 0.85
            sf.write(str(cached_wav), y, sr, subtype="PCM_16")
        except Exception as e:
            # Fallback nếu mất mạng: dùng gTTS
            from gtts import gTTS
            tts = gTTS(text=text, lang="vi", slow=False)
            tts.save(tmp_mp3)
            y, _ = librosa.load(tmp_mp3, sr=sr, mono=True)
            y = y / (np.max(np.abs(y)) + 1e-8) * 0.85
            sf.write(str(cached_wav), y, sr, subtype="PCM_16")
        finally:
            if os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

    duration_sec = len(y) / sr
    rtf = (inference_time_ms / 1000.0) / max(0.01, duration_sec)

    # 2. Trích xuất Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=50,
        fmax=7600
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    extra = {
        "text": text,
        "latency_ms": round(inference_time_ms, 2),
        "duration_sec": round(duration_sec, 3),
        "rtf": round(rtf, 4),
        "mel_spectrogram": mel_db,
        "cached_path": str(cached_wav).replace("\\", "/")
    }
    return y.astype(np.float32), sr, extra


if __name__ == "__main__":
    test_text = "Mời khách hàng số 23 đến quầy 5."
    y, sr, extra = synthesize(test_text)
    print(f"[+] Neural TTS hoàn tất: {extra['duration_sec']}s, Latency: {extra['latency_ms']}ms, RTF: {extra['rtf']}")
