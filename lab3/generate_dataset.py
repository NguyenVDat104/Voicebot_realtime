"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 3
Mô-đun Tổng hợp Bộ Dữ liệu Chuẩn 5 Lệnh Tiếng Việt (Dataset Generator)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Tập lệnh: 5 lệnh tách rời: ["mở", "đóng", "lên", "xuống", "dừng"]
Cấu trúc:
  - Train: 6 utterance / lớp (dataset/train/<label>/<label>_01.wav ... 06.wav)
  - Test: 2 utterance / lớp (dataset/test/<label>/<label>_07.wav, 08.wav)
  - Mismatch Speaker 2: 2 utterance / lớp (dataset/test_speaker2/<label>/...)
Quy chuẩn:
  - Sampling Rate: 16,000 Hz, Mono, PCM 16-bit.
  - VAD Silence Padding: 0.25s - 0.45s ở đầu và cuối.
===============================================================================
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS

VOCAB = {
    'mo': 'mở',
    'dong': 'đóng',
    'len': 'lên',
    'xuong': 'xuống',
    'dung': 'dừng'
}

TARGET_FS = 16000

# 8 biến thể tự nhiên cho Speaker 1 (6 train + 2 test)
# Các tham số mô phỏng tốc độ nói (stretch: >1 là nhanh hơn, <1 là chậm hơn)
# và cao độ (pitch_steps: số nửa cung / semitone)
SPK1_VARIATIONS = [
    # Train: 01 -> 06
    {'stretch': 1.00, 'pitch_steps': 0.0,  'lead_s': 0.30, 'trail_s': 0.35},
    {'stretch': 0.94, 'pitch_steps': -0.7, 'lead_s': 0.25, 'trail_s': 0.30},
    {'stretch': 1.06, 'pitch_steps': 0.8,  'lead_s': 0.35, 'trail_s': 0.28},
    {'stretch': 0.90, 'pitch_steps': -1.2, 'lead_s': 0.40, 'trail_s': 0.38},
    {'stretch': 1.08, 'pitch_steps': 0.6,  'lead_s': 0.28, 'trail_s': 0.42},
    {'stretch': 0.97, 'pitch_steps': 0.3,  'lead_s': 0.32, 'trail_s': 0.32},
    # Test: 07 -> 08 (độc lập hoàn toàn với train)
    {'stretch': 1.04, 'pitch_steps': -0.5, 'lead_s': 0.33, 'trail_s': 0.34},
    {'stretch': 0.93, 'pitch_steps': 1.0,  'lead_s': 0.27, 'trail_s': 0.40},
]

# 2 biến thể cho Speaker 2 (Cross-speaker test: cao độ nâng lên +3.5 semitones mô phỏng giọng nữ/khác người nói)
SPK2_VARIATIONS = [
    {'stretch': 1.05, 'pitch_steps': 3.5,  'lead_s': 0.30, 'trail_s': 0.35},
    {'stretch': 0.95, 'pitch_steps': 4.2,  'lead_s': 0.36, 'trail_s': 0.30},
]


def generate_base_audio(text, tmp_mp3):
    """Sinh âm thanh gốc bằng gTTS tiếng Việt."""
    tts = gTTS(text=text, lang='vi', slow=False)
    tts.save(tmp_mp3)


def transform_and_save(input_mp3, output_wav, stretch=1.0, pitch_steps=0.0, lead_s=0.3, trail_s=0.35, target_fs=TARGET_FS):
    """
    Áp dụng biến thiên tốc độ, cao độ, thêm khoảng lặng đầu/cuối
    và xuất định dạng chuẩn 16-bit PCM WAV ở 16kHz.
    """
    y, sr = librosa.load(input_mp3, sr=target_fs, mono=True)

    # Cắt khoảng lặng ban đầu từ TTS engine nếu có
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    # Biến thiên tốc độ (Time stretch)
    if stretch != 1.0:
        y_mod = librosa.effects.time_stretch(y_trimmed, rate=stretch)
    else:
        y_mod = y_trimmed

    # Biến thiên cao độ (Pitch shift)
    if pitch_steps != 0.0:
        y_mod = librosa.effects.pitch_shift(y_mod, sr=target_fs, n_steps=pitch_steps)

    # Chuẩn hóa biên độ tín hiệu tiếng nói về max 0.90
    max_val = np.max(np.abs(y_mod))
    if max_val > 0:
        y_mod = y_mod / max_val * 0.90

    # Tạo vùng đệm khoảng lặng (silence buffer) ở hai đầu
    lead_len = int(lead_s * target_fs)
    trail_len = int(trail_s * target_fs)

    noise_level = 1e-4  # Thêm dither siêu nhỏ chống lỗi chia cho 0
    rng = np.random.default_rng(42)
    lead_noise = rng.normal(0, noise_level, lead_len).astype(np.float32)
    trail_noise = rng.normal(0, noise_level, trail_len).astype(np.float32)

    full_audio = np.concatenate([lead_noise, y_mod.astype(np.float32), trail_noise])

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, full_audio, target_fs, subtype='PCM_16')


def generate_all_datasets():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    spk2_dir = dataset_dir / "test_speaker2"

    print("=" * 65)
    print("BẮT ĐẦU TỔNG HỢP BỘ DỮ LIỆU LAB 3 (5 LỆNH TIẾNG VIỆT)")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Bước A: Sinh âm thanh gốc cho 5 từ
        base_audios = {}
        for label, text in VOCAB.items():
            tmp_mp3 = os.path.join(tmp_dir, f"{label}_base.mp3")
            generate_base_audio(text, tmp_mp3)
            base_audios[label] = tmp_mp3
            print(f"  Đã tạo base TTS cho lệnh: [{label}] -> '{text}'")

        # Bước B: Sinh tập Speaker 1 (Train: 01-06, Test: 07-08)
        print(f"\n[1/2] Đang tạo dữ liệu Speaker 1 (Baseline)...")
        for label in VOCAB:
            tmp_mp3 = base_audios[label]
            for i, var in enumerate(SPK1_VARIATIONS, start=1):
                if i <= 6:
                    out_wav = train_dir / label / f"{label}_{i:02d}.wav"
                    split_tag = "TRAIN"
                else:
                    out_wav = test_dir / label / f"{label}_{i:02d}.wav"
                    split_tag = "TEST "

                transform_and_save(
                    tmp_mp3, out_wav,
                    stretch=var['stretch'],
                    pitch_steps=var['pitch_steps'],
                    lead_s=var['lead_s'],
                    trail_s=var['trail_s']
                )
                info = sf.info(out_wav)
                print(f"  [{split_tag}] {label}/{out_wav.name} | {info.samplerate}Hz | {info.duration:.2f}s | {info.subtype}")

        # Bước C: Sinh tập Speaker 2 (Cross-speaker mismatch)
        print(f"\n[2/2] Đang tạo dữ liệu Speaker 2 (Cross-Speaker Mismatch)...")
        for label in VOCAB:
            tmp_mp3 = base_audios[label]
            for i, var in enumerate(SPK2_VARIATIONS, start=1):
                out_wav = spk2_dir / label / f"{label}_spk2_{i:02d}.wav"
                transform_and_save(
                    tmp_mp3, out_wav,
                    stretch=var['stretch'],
                    pitch_steps=var['pitch_steps'],
                    lead_s=var['lead_s'],
                    trail_s=var['trail_s']
                )
                info = sf.info(out_wav)
                print(f"  [SPK2 ] {label}/{out_wav.name} | {info.samplerate}Hz | {info.duration:.2f}s | {info.subtype}")

    print("\n" + "=" * 65)
    print("HOÀN TẤT SINH BỘ DỮ LIỆU:")
    print(f" - Train (6/từ x 5 = 30 files): {train_dir}")
    print(f" - Test  (2/từ x 5 = 10 files): {test_dir}")
    print(f" - Test Speaker 2 (10 files)  : {spk2_dir}")
    print("=" * 65)


if __name__ == '__main__':
    generate_all_datasets()
