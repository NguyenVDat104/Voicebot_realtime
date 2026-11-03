"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4
Công cụ Thu âm Kho Dữ liệu Inventory Trực tiếp qua Microphone (Voice Recorder)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Mục đích:
  Cho phép sinh viên tự thu âm bằng giọng thật qua microphone cho từng từ vựng
  trong bài toán "Thông báo xếp hàng" và tự động chuẩn hóa đạt chuẩn Lab 4.
===============================================================================
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import pandas as pd

# Danh sách từ vựng cần thu
VOCAB_UNITS = [
    ("moi", "mời"),
    ("khach", "khách"),
    ("hang", "hàng"),
    ("so", "số"),
    ("den", "đến"),
    ("quay", "quầy"),
    ("khong", "không"),
    ("mot", "một"),
    ("hai", "hai"),
    ("ba", "ba"),
    ("bon", "bốn"),
    ("nam", "năm"),
    ("sau", "sáu"),
    ("bay", "bảy"),
    ("tam", "tám"),
    ("chin", "chín"),
]

TARGET_SR = 16000
TARGET_RMS = 0.08
RECORD_SECONDS = 1.8  # Mỗi từ nói tách rời ~1.5 - 2s là đủ


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def match_rms(x, target=TARGET_RMS):
    current_rms = rms(x)
    if current_rms < 1e-6:
        return x
    return np.clip(x * (target / current_rms), -0.99, 0.99)


def record_word(prompt_word, duration=RECORD_SECONDS, sr=TARGET_SR):
    """Thu âm một từ qua microphone."""
    print(f"\n>>> Chuẩn bị phát âm từ: [ {prompt_word.upper()} ]")
    input("    Nhấn ENTER để bắt đầu thu âm... ")
    print(f"    [*] ĐANG THU ÂM ({duration}s)... Hãy nói rõ ràng: '{prompt_word}'")
    
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("    [+] Đã hoàn thành đoạn thu.")
    return audio.flatten()


def process_and_save(audio, output_path, sr=TARGET_SR):
    """Cắt silence, pad 15ms và match RMS."""
    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=25, frame_length=512, hop_length=128)
    if len(trimmed) == 0:
        trimmed = audio

    # Padding 15ms
    pad = int(sr * 0.015)
    padded = np.pad(trimmed, (pad, pad), mode="constant")

    # Match RMS
    final_audio = match_rms(padded, target=TARGET_RMS)
    sf.write(str(output_path), final_audio, sr, subtype="PCM_16")
    dur = len(final_audio) / sr
    cur_rms = rms(final_audio)
    print(f"    -> Đã lưu: {output_path.name} ({dur:.2f}s, RMS={cur_rms:.4f})")
    return dur, cur_rms


def interactive_recording():
    base_dir = Path(__file__).resolve().parent
    units_dir = base_dir / "lab4_data" / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" CÔNG CỤ THU ÂM INVENTORY LAB 4 - CSE457 (THÔNG BÁO XẾP HÀNG)")
    print(" Chuẩn: Fs = 16,000 Hz, Mono, Tự động trim silence và match RMS")
    print("=" * 70)
    print("Lựa chọn chế độ:")
    print(" 1. Thu toàn bộ 16 từ (mỗi từ thu candidate 01 và 02)")
    print(" 2. Thu bổ sung/ghi đè 1 từ cụ thể")
    print(" 0. Thoát")
    
    choice = input("Nhập lựa chọn của bạn (1/2/0): ").strip()
    if choice == "0":
        print("Tạm biệt!")
        return

    if choice == "1":
        for slug, word in VOCAB_UNITS:
            for cand in ["01", "02"]:
                out_path = units_dir / f"{slug}_{cand}.wav"
                raw = record_word(f"{word} (Lần thu {cand})")
                process_and_save(raw, out_path)
        print("\n[+] Đã hoàn thành thu âm toàn bộ kho từ vựng!")

    elif choice == "2":
        print("\nDanh sách từ:")
        for idx, (slug, word) in enumerate(VOCAB_UNITS):
            print(f" {idx+1:2d}. {word} ({slug})")
        sel = int(input("\nChọn số thứ tự từ muốn thu: ")) - 1
        slug, word = VOCAB_UNITS[sel]
        cand = input("Nhập candidate ID (ví dụ 01, 02 hoặc 03): ").strip() or "01"
        out_path = units_dir / f"{slug}_{cand}.wav"
        raw = record_word(f"{word} (Lần thu {cand})")
        process_and_save(raw, out_path)


if __name__ == "__main__":
    interactive_recording()
