"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4
Bộ Tiền Xử lý Dữ liệu Thu âm Thực tế (Raw Recordings Processor)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Nhiệm vụ:
  1. Đọc toàn bộ file thu âm thực tế từ thư mục: lab4/raw_recording/<từ>/
  2. Tự động phát hiện tiếng nói (Energy-based VAD), cắt bỏ silence đầu/cuối sạch sẽ.
  3. Padding 20ms để bảo toàn phụ âm bật/xát.
  4. Đảm bảo Sampling Rate chuẩn 16,000 Hz, Mono, PCM 16-bit.
  5. Cân bằng năng lượng chuẩn hóa RMS = 0.08 theo đề bài.
  6. Đồng bộ vào kho: lab4_data/units/<từ>_<candidate>.wav và cập nhật metadata.csv.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_recording"
UNITS_DIR = BASE_DIR / "lab4_data" / "units"
FIGURES_DIR = BASE_DIR / "figures"

UNITS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
TARGET_RMS = 0.08

WORD_MAP = {
    "moi": "mời", "khach": "khách", "hang": "hàng", "so": "số",
    "den": "đến", "quay": "quầy",
    "khong": "không", "mot": "một", "hai": "hai", "ba": "ba", "bon": "bốn",
    "nam": "năm", "sau": "sáu", "bay": "bảy", "tam": "tám", "chin": "chín"
}


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def match_rms(x, target=TARGET_RMS):
    cur_rms = rms(x)
    if cur_rms < 1e-6:
        return x
    return np.clip(x * (target / cur_rms), -0.99, 0.99)


def trim_speech(y, sr=TARGET_SR, frame_ms=30, pad_ms=20):
    """
    Cắt silence thích nghi dựa trên đường bao năng lượng (Energy-based VAD)
    giúp tách chính xác đoạn phát âm thật, loại bỏ tiếng thở/nhiễu phòng.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    energies = np.array([
        np.sqrt(np.mean(y[i : i + frame_len] ** 2))
        for i in range(0, len(y) - frame_len, hop)
    ])
    max_e = np.max(energies)
    thresh = max(0.045, max_e * 0.22)
    voiced = np.where(energies > thresh)[0]
    if len(voiced) == 0:
        return y

    start_frame = voiced[0]
    end_frame = voiced[-1]
    start_sample = max(0, start_frame * hop - int(sr * pad_ms / 1000))
    end_sample = min(len(y), (end_frame + 2) * hop + int(sr * pad_ms / 1000))
    return y[start_sample:end_sample]


def process_raw_dataset():
    print("=" * 75)
    print(" TIỀN XỬ LÝ VÀ CHUẨN HÓA BỘ DỮ LIỆU THU ÂM THỰC TẾ LAB 4")
    print(f" Thư mục nguồn: {RAW_DIR}")
    print(f" Thư mục đích:   {UNITS_DIR}")
    print("=" * 75)

    metadata_records = []
    processed_count = 0

    for slug, vn_label in WORD_MAP.items():
        word_dir = RAW_DIR / slug
        if not word_dir.exists():
            print(f"[!] Cảnh báo: Không tìm thấy thư mục {word_dir.name}")
            continue

        # Lấy danh sách file wav trong thư mục
        wav_files = sorted([f for f in word_dir.iterdir() if f.suffix.lower() == ".wav"])
        # Ưu tiên lấy 3 candidate: nếu có _01, _02, _03 thì lấy, nếu không lấy 3 file đầu
        cands_to_take = []
        for cid in ["01", "02", "03"]:
            target_f = word_dir / f"{slug}_{cid}.wav"
            if target_f.exists():
                cands_to_take.append((cid, target_f))
        if len(cands_to_take) < 3:
            # Fallback lấy các file khác nếu thiếu
            existing_cids = [c[0] for c in cands_to_take]
            for f in wav_files:
                if len(cands_to_take) >= 3:
                    break
                # Trích xuất hoặc gán ID
                cid_fallback = f"0{len(cands_to_take)+1}"
                if f not in [c[1] for c in cands_to_take]:
                    cands_to_take.append((cid_fallback, f))

        print(f"\n[*] Đang xử lý từ '{vn_label}' ({slug}): {len(cands_to_take)} candidates")
        for cid, src_file in cands_to_take:
            y, sr = sf.read(str(src_file), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

            raw_dur = len(y) / TARGET_SR
            trimmed_y = trim_speech(y, TARGET_SR)
            norm_y = match_rms(trimmed_y, target=TARGET_RMS)
            final_dur = len(norm_y) / TARGET_SR

            out_filename = f"{slug}_{cid}.wav"
            out_path = UNITS_DIR / out_filename
            sf.write(str(out_path), norm_y, TARGET_SR, subtype="PCM_16")

            metadata_records.append({
                "unit_slug": slug,
                "unit_label": vn_label,
                "candidate_id": cid,
                "filename": out_filename,
                "duration_sec": round(final_dur, 4),
                "samples": len(norm_y),
                "rms": round(rms(norm_y), 6),
                "file_path": str(out_path).replace("\\", "/")
            })
            processed_count += 1
            print(f"    - [{cid}] {src_file.name} (gốc: {raw_dur:.2f}s) -> {out_filename} ({final_dur:.2f}s, RMS={rms(norm_y):.4f})")

    # Lưu metadata.csv
    df_meta = pd.DataFrame(metadata_records)
    meta_path = BASE_DIR / "lab4_data" / "metadata.csv"
    df_meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Đã cập nhật metadata: {meta_path} ({len(df_meta)} mẫu)")

    # Vẽ lại fig1_inventory_stats.png theo giọng thật
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].hist(df_meta["duration_sec"], bins=10, color="#2b5c8f", edgecolor="black", alpha=0.85)
    mean_d = df_meta["duration_sec"].mean()
    axes[0].axvline(mean_d, color="red", linestyle="--", label=f"Mean: {mean_d:.2f}s")
    axes[0].set_title("Phân bố Thời lượng Unit Thực tế (Giọng thật)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Thời lượng (giây)")
    axes[0].set_ylabel("Số lượng mẫu")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].hist(df_meta["rms"], bins=10, color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[1].axvline(TARGET_RMS, color="red", linestyle="--", label=f"Target: {TARGET_RMS:.2f}")
    axes[1].set_title("Phân bố Năng lượng RMS sau Chuẩn hóa", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("RMS")
    axes[1].set_ylabel("Số lượng mẫu")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_inventory_stats.png")
    plt.close()
    print(f"[+] Đã cập nhật biểu đồ: {FIGURES_DIR / 'fig1_inventory_stats.png'}")
    return df_meta


if __name__ == "__main__":
    process_raw_dataset()
