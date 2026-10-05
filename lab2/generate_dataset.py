"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 2
Mô-đun Thu thập và Tổng hợp Bộ Dữ liệu Mẫu (Dataset Generator)

Tệp: generate_dataset.py
Mục đích:
  - Tự động sinh tập dữ liệu 5 từ tiếng Việt: ["không", "một", "hai", "ba", "bốn"]
  - Tạo tối thiểu 5 lần lặp cho mỗi từ (tổng cộng 25 file cho Speaker 1).
  - Tạo thêm 25 file cho Speaker 2 (nữ) để phục vụ thí nghiệm mở rộng Cross-Speaker.
  - Chuẩn hóa nghiêm ngặt theo đặc tả đề bài Lab 2:
      + Định dạng: WAV, Mono, PCM 16-bit, Sampling rate Fs = 16,000 Hz.
      + Có khoảng lặng (silence buffer) 0.25s - 0.4s ở đầu và cuối mỗi từ
        nhằm kiểm tra và đánh giá chính xác thuật toán Endpoint Detection.
      + Tạo biến thiên tự nhiên về tốc độ nói (rate) và cao độ (pitch) giữa các lần lặp.
===============================================================================
"""

import os
import sys
import asyncio
import tempfile
import numpy as np
import soundfile as sf
import librosa

# Khắc phục lỗi SSL trên Windows khi aiohttp duyệt chứng chỉ hệ thống
import ssl
import certifi
try:
    ssl.SSLContext.load_default_certs = lambda self, purpose=None: self.load_verify_locations(cafile=certifi.where())
except Exception as e:
    pass

import edge_tts

# Các từ vựng tiếng Việt và nhãn thư mục ASCII chuẩn hóa
VOCAB = {
    'khong': 'không',
    'mot': 'một',
    'hai': 'hai',
    'ba': 'ba',
    'bon': 'bốn'
}

TARGET_FS = 16000

# Cấu hình biến thiên tốc độ và cao độ cho 5 lần phát âm (mô phỏng người nói tự nhiên)
VARIATIONS = [
    {'rate': '+0%',  'pitch': '+0Hz',  'lead_s': 0.30, 'trail_s': 0.35},
    {'rate': '-6%',  'pitch': '-5Hz',  'lead_s': 0.25, 'trail_s': 0.30},
    {'rate': '+6%',  'pitch': '+6Hz',  'lead_s': 0.35, 'trail_s': 0.28},
    {'rate': '-10%', 'pitch': '-3Hz',  'lead_s': 0.40, 'trail_s': 0.38},
    {'rate': '+8%',  'pitch': '+4Hz',  'lead_s': 0.28, 'trail_s': 0.42},
]


async def synthesize_word(text, voice, rate, pitch, output_mp3):
    """Tổng hợp phát âm 1 từ qua Edge-TTS với biến thiên rate và pitch."""
    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await comm.save(output_mp3)


def process_and_save_wav(input_mp3, output_wav, target_fs=TARGET_FS, lead_s=0.3, trail_s=0.35):
    """
    Đọc MP3, chuyển đổi sang mono, resample về 16kHz,
    thêm khoảng lặng ở hai đầu và lưu dưới dạng 16-bit PCM WAV.
    """
    data, sr = sf.read(input_mp3)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample về chuẩn 16,000 Hz nếu khác sampling rate
    if sr != target_fs:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_fs)

    # Chuẩn hóa biên độ tín hiệu tiếng nói về max 0.90
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.90

    # Tạo khoảng lặng đầu và cuối có thêm một chút nhiễu phòng siêu nhỏ (dither)
    lead_len = int(lead_s * target_fs)
    trail_len = int(trail_s * target_fs)
    
    noise_level = 1e-4  # Nhiễu nền siêu nhỏ tránh lỗi chia 0
    rng = np.random.default_rng(42)
    lead_noise = rng.normal(0, noise_level, lead_len).astype(np.float32)
    trail_noise = rng.normal(0, noise_level, trail_len).astype(np.float32)

    # Ghép chuỗi âm thanh: [Khoảng lặng đầu] + [Tiếng nói] + [Khoảng lặng cuối]
    full_audio = np.concatenate([lead_noise, data.astype(np.float32), trail_noise])

    # Đảm bảo thư mục cha tồn tại
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    # Ghi file PCM 16-bit WAV
    sf.write(output_wav, full_audio, target_fs, subtype='PCM_16')


async def generate_dataset_for_speaker(speaker_name, voice_name, base_folder):
    """Sinh toàn bộ 25 file cho một người nói."""
    print(f"\n=======================================================")
    print(f"Đang sinh bộ dữ liệu cho: {speaker_name} (Voice: {voice_name})")
    print(f"Thư mục lưu trữ: {base_folder}")
    print(f"=======================================================")

    count = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for lab, text in VOCAB.items():
            lab_dir = os.path.join(base_folder, lab)
            os.makedirs(lab_dir, exist_ok=True)

            for i, var in enumerate(VARIATIONS, start=1):
                file_name = f"{lab}_{i:02d}.wav"
                out_wav = os.path.join(lab_dir, file_name)
                tmp_mp3 = os.path.join(tmp_dir, f"{lab}_{i}.mp3")

                # Tổng hợp âm thanh
                await synthesize_word(
                    text=text,
                    voice=voice_name,
                    rate=var['rate'],
                    pitch=var['pitch'],
                    output_mp3=tmp_mp3
                )

                # Chuẩn hóa về 16kHz mono WAV với silence padding
                process_and_save_wav(
                    input_mp3=tmp_mp3,
                    output_wav=out_wav,
                    target_fs=TARGET_FS,
                    lead_s=var['lead_s'],
                    trail_s=var['trail_s']
                )

                count += 1
                # Kiểm tra thông tin file vừa tạo
                info = sf.info(out_wav)
                print(f"  [{count:02d}/25] Đã tạo: {file_name} | {info.samplerate}Hz | {info.duration:.2f}s | {info.subtype}")

    print(f"--> Hoàn tất sinh {count} file cho {speaker_name} tại {base_folder}!")


async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Speaker 1: Giọng nam (vi-VN-NamMinhNeural) - Dataset chính của bài Lab
    spk1_dir = os.path.join(base_dir, "dataset")
    await generate_dataset_for_speaker("Người nói 1 (Nam - Baseline)", "vi-VN-NamMinhNeural", spk1_dir)

    # Speaker 2: Giọng nữ (vi-VN-HoaiMyNeural) - Phục vụ thí nghiệm Cross-Speaker
    spk2_dir = os.path.join(base_dir, "dataset_speaker2")
    await generate_dataset_for_speaker("Người nói 2 (Nữ - Cross-Speaker)", "vi-VN-HoaiMyNeural", spk2_dir)

    print("\n" + "=" * 60)
    print("TỔNG HỢP DATASET LAB 2 THÀNH CÔNG RỰC RỠ!")
    print(f"1. Bộ dữ liệu chính (Speaker 1): {spk1_dir}")
    print(f"2. Bộ dữ liệu đối chứng (Speaker 2): {spk2_dir}")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
