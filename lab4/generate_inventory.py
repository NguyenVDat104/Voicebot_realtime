"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4
Mô-đun Khởi tạo & Chuẩn hóa Kho Dữ liệu Inventory (Case Study: Thông báo xếp hàng)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Tập từ vựng mục tiêu (16 từ vựng):
  - Nhóm điều khiển (6 từ): "mời", "khách", "hàng", "số", "đến", "quầy"
  - Nhóm chữ số (10 từ): "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"

Quy chuẩn đầu ra:
  - Sampling Rate: 16,000 Hz, Mono, Float32 / PCM 16-bit
  - Cắt silence đầu/cuối sạch (padding ~20ms)
  - Chuẩn hóa RMS: match_rms(x, target=0.08)
  - Số lượng: 3 candidates / từ vựng (tổng cộng 48 file WAV)
===============================================================================
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import pandas as pd

# Danh mục từ vựng (Slug -> Tiếng Việt có dấu)
VOCAB_UNITS = {
    # Điều khiển
    "moi": "mời",
    "khach": "khách",
    "hang": "hàng",
    "so": "số",
    "den": "đến",
    "quay": "quầy",
    # Chữ số 0 - 9
    "khong": "không",
    "mot": "một",
    "hai": "hai",
    "ba": "ba",
    "bon": "bốn",
    "nam": "năm",
    "sau": "sáu",
    "bay": "bảy",
    "tam": "tám",
    "chin": "chín",
}

TARGET_SR = 16000
TARGET_RMS = 0.08

# Cấu hình biến thiên cho 3 candidate nhằm phục vụ thuật toán Unit Selection
CANDIDATE_CONFIGS = [
    {"id": "01", "rate": "+0%",  "pitch": "+0Hz",  "stretch": 1.00, "pitch_shift": 0.0},
    {"id": "02", "rate": "-4%",  "pitch": "-3Hz",  "stretch": 0.96, "pitch_shift": -0.6},
    {"id": "03", "rate": "+4%",  "pitch": "+4Hz",  "stretch": 1.04, "pitch_shift": 0.7},
]


def rms(x):
    """Tính Root Mean Square của tín hiệu."""
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def match_rms(x, target=TARGET_RMS):
    """Chuẩn hóa mức năng lượng RMS theo đúng định dạng Lab 4."""
    current_rms = rms(x)
    if current_rms < 1e-6:
        return x
    y = x * (target / current_rms)
    return np.clip(y, -0.99, 0.99)


async def generate_single_edge_tts(text, output_path, rate="+0%", pitch="+0Hz", voice="vi-VN-HoaiMyNeural"):
    """Tổng hợp giọng nói tiếng Việt bằng edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def synthesize_word_audio(text, candidate_cfg):
    """
    Sinh audio gốc cho 1 từ với biến thiên nhẹ để tạo candidate chân thực.
    Ưu tiên edge-tts (chất lượng tự nhiên), fallback gTTS nếu không có kết nối.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_f:
        tmp_mp3 = tmp_f.name

    try:
        # Thử với edge-tts
        asyncio.run(generate_single_edge_tts(
            text=text,
            output_path=tmp_mp3,
            rate=candidate_cfg["rate"],
            pitch=candidate_cfg["pitch"]
        ))
        x, fs = librosa.load(tmp_mp3, sr=TARGET_SR, mono=True)
    except Exception as e:
        # Fallback với gTTS
        from gtts import gTTS
        tts = gTTS(text=text, lang="vi", slow=False)
        tts.save(tmp_mp3)
        x, fs = librosa.load(tmp_mp3, sr=TARGET_SR, mono=True)

        # Biến thiên hậu kỳ bằng librosa
        if candidate_cfg["stretch"] != 1.0:
            x = librosa.effects.time_stretch(x, rate=candidate_cfg["stretch"])
        if candidate_cfg["pitch_shift"] != 0.0:
            x = librosa.effects.pitch_shift(x, sr=TARGET_SR, n_steps=candidate_cfg["pitch_shift"])
    finally:
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)

    # 1. Cắt silence đầu và cuối (trim) với top_db hợp lý (~28dB)
    trimmed, _ = librosa.effects.trim(x, top_db=28, frame_length=512, hop_length=128)

    # 2. Thêm một đoạn padding nhỏ (~15ms) ở đầu và cuối để tránh cắt cụt âm
    pad_samples = int(TARGET_SR * 0.015)
    padded = np.pad(trimmed, (pad_samples, pad_samples), mode="constant")

    # 3. Chuẩn hóa RMS theo mục tiêu Lab 4
    normalized = match_rms(padded, target=TARGET_RMS)
    return normalized.astype(np.float32)


def build_inventory(base_dir):
    """Xây dựng và lưu trữ toàn bộ kho âm thanh Inventory."""
    units_dir = base_dir / "lab4_data" / "units"
    figures_dir = base_dir / "figures"
    units_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    total_words = len(VOCAB_UNITS)
    total_candidates = len(CANDIDATE_CONFIGS)
    total_files = total_words * total_candidates

    print(f"[*] Bắt đầu sinh kho dữ liệu Inventory: {total_words} từ vựng x {total_candidates} candidates = {total_files} file WAV.")
    idx = 0

    for slug, vietnamese_word in VOCAB_UNITS.items():
        for cand_cfg in CANDIDATE_CONFIGS:
            idx += 1
            cand_id = cand_cfg["id"]
            filename = f"{slug}_{cand_id}.wav"
            file_path = units_dir / filename

            # Sinh audio và chuẩn hóa
            audio = synthesize_word_audio(vietnamese_word, cand_cfg)
            sf.write(str(file_path), audio, TARGET_SR, subtype="PCM_16")

            duration = len(audio) / TARGET_SR
            unit_rms = rms(audio)

            metadata_records.append({
                "unit_slug": slug,
                "unit_label": vietnamese_word,
                "candidate_id": cand_id,
                "filename": filename,
                "duration_sec": round(duration, 4),
                "samples": len(audio),
                "rms": round(unit_rms, 6),
                "file_path": str(file_path).replace("\\", "/")
            })
            print(f"  [{idx:02d}/{total_files:02d}] '{vietnamese_word}' ({cand_id}) -> {filename} ({duration:.2f}s, RMS={unit_rms:.4f})")

    # Lưu metadata.csv
    df_meta = pd.DataFrame(metadata_records)
    metadata_csv_path = base_dir / "lab4_data" / "metadata.csv"
    df_meta.to_csv(metadata_csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] Đã lưu bảng đặc tả metadata: {metadata_csv_path}")

    # Vẽ biểu đồ kiểm tra phân phối dữ liệu (Yêu cầu 4.1.A)
    plot_inventory_stats(df_meta, figures_dir / "fig1_inventory_stats.png")
    return df_meta


def plot_inventory_stats(df, save_path):
    """Vẽ histogram duration và RMS để kiểm tra tính đồng đều dữ liệu (Yêu cầu 4.1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Subplot 1: Duration Histogram
    ax1 = axes[0]
    ax1.hist(df["duration_sec"], bins=10, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax1.set_title("Phân bố Thời lượng Unit (Duration Distribution)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Thời lượng (giây)")
    ax1.set_ylabel("Số lượng mẫu")
    ax1.grid(True, linestyle="--", alpha=0.5)
    mean_dur = df["duration_sec"].mean()
    ax1.axvline(mean_dur, color="red", linestyle="--", label=f"Mean: {mean_dur:.2f}s")
    ax1.legend()

    # Subplot 2: RMS Energy Histogram
    ax2 = axes[1]
    ax2.hist(df["rms"], bins=10, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_title("Phân bố Năng lượng RMS sau Chuẩn hóa", fontsize=11, fontweight="bold")
    ax2.set_xlabel("RMS")
    ax2.set_ylabel("Số lượng mẫu")
    ax2.grid(True, linestyle="--", alpha=0.5)
    mean_rms = df["rms"].mean()
    ax2.axvline(mean_rms, color="red", linestyle="--", label=f"Target: {TARGET_RMS:.2f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Đã lưu biểu đồ phân tích Inventory: {save_path}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    build_inventory(current_dir)
