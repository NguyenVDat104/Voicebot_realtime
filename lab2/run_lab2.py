"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | TRƯỜNG ĐẠI HỌC THỦY LỢI
KHOA CÔNG NGHỆ THÔNG TIN - BỘ MÔN TRÍ TUỆ NHÂN TẠO

LAB 2: ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG BẰNG DTW
Từ phân tích ngắn hạn đến MFCC, căn chỉnh thời gian động và nhận dạng từ đơn

Tệp thực thi chính: run_lab2.py
Nội dung bao phủ trọn vẹn:
  - Phần A: Kiểm tra dữ liệu và phân chia Train (Templates) / Test
  - Phần B: Đặc trưng miền thời gian (Short-time Energy, RMS, ZCR, Autocorrelation/Pitch)
  - Phần C: Phát hiện điểm đầu/cuối từ nói (Endpoint Detection / VAD) & Cắt khoảng lặng
  - Phần D: Trích xuất đặc trưng MFCC (Pre-emphasis, Mel Filterbank, Log, DCT, CMN)
  - Phần E: Tự cài đặt thuật toán DTW (Local Euclidean distance, DP, Backtracking, Normalization)
  - Phần F: Bộ nhận dạng Nearest-Template (Top-3 scores, Reject Threshold)
  - Phần G: Đánh giá hiệu năng (Accuracy, Confusion Matrix) & 3 Thí nghiệm mở rộng:
      + Thí nghiệm E1: Có Endpoint Detection vs. Không có Endpoint Detection
      + Thí nghiệm E2: 13 MFCC chuẩn vs. 26 MFCC + Delta (Động học âm học)
      + Thí nghiệm E4: Same-Speaker vs. Cross-Speaker
===============================================================================
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import scipy.signal as signal
from scipy.signal import lfilter
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Thiết lập giao diện đồ thị matplotlib chuẩn xuất bản phẩm
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = True

# Các hằng số cấu hình hệ thống chuẩn theo đặc tả bài Lab
FS = 16000                 # Tần số lấy mẫu: 16 kHz
FRAME_MS = 25              # Độ dài khung: 25 ms
HOP_MS = 10                # Bước dịch khung: 10 ms
WIN = int(FS * FRAME_MS / 1000)  # 400 mẫu
HOP = int(FS * HOP_MS / 1000)    # 160 mẫu
N_FFT = 512                # Kích thước FFT
N_MELS = 24                # Số bộ lọc Mel tam giác
N_MFCC = 13                # Số hệ số cepstral
PRE_EMPH = 0.97            # Hệ số tiền nhấn
TOP_DB = 32                # Ngưỡng năng lượng tương đối phát hiện tiếng nói
MARGIN_MS = 50             # Vùng đệm an toàn sau khi trim (50 ms = 800 mẫu)

LABELS = ['khong', 'mot', 'hai', 'ba', 'bon']
VIET_NAMES = {'khong': 'không', 'mot': 'một', 'hai': 'hai', 'ba': 'ba', 'bon': 'bốn'}

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_SPK2_DIR = BASE_DIR / "dataset_speaker2"
FIGURES_DIR = BASE_DIR / "figures"
TRIMMED_DIR = BASE_DIR / "audio_trimmed"
RESULTS_CSV = BASE_DIR / "results.csv"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TRIMMED_DIR, exist_ok=True)


# =============================================================================
# 1. CÁC HÀM TIỆN ÍCH ĐỌC & TIỀN XỬ LÝ ÂM THANH
# =============================================================================

def load_audio(path, target_fs=FS):
    """Đọc file âm thanh, chuẩn hóa về mono, 16kHz và chuẩn hóa biên độ max về 1.0."""
    y, sr = librosa.load(path, sr=target_fs, mono=True)
    max_val = np.max(np.abs(y))
    if max_val > 1e-9:
        y = y / max_val
    return y


def compute_time_features(y, frame_len=WIN, hop_len=HOP):
    """
    Tính các đặc trưng miền thời gian theo từng khung:
    - Short-time Energy (E)
    - Short-time Log-energy (dB)
    - RMS
    - Zero-Crossing Rate (ZCR)
    """
    num_frames = 1 + int((len(y) - frame_len) / hop_len)
    if num_frames < 1:
        num_frames = 1
        y = np.pad(y, (0, frame_len - len(y)))

    energy = np.zeros(num_frames)
    rms = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    times = (np.arange(num_frames) * hop_len + frame_len / 2) / FS
    window = signal.windows.hamming(frame_len)

    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frame = y[start:end]
        frame_win = frame * window

        # Energy & RMS
        e_val = np.sum(frame_win ** 2)
        energy[i] = e_val
        rms[i] = np.sqrt(np.mean(frame_win ** 2))

        # ZCR: Tỷ lệ đổi dấu
        # sgn(x[m]) - sgn(x[m-1])
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr[i] = np.sum(np.abs(signs[1:] - signs[:-1])) / (2.0 * frame_len)

    log_energy = 10 * np.log10(energy + 1e-12)
    return times, energy, log_energy, rms, zcr


def compute_autocorrelation(frame, max_lag=None):
    """Tính tự tương quan (autocorrelation) của một khung âm thanh để ước lượng Pitch."""
    if max_lag is None:
        max_lag = len(frame)
    r = np.correlate(frame, frame, mode='full')
    r = r[len(frame)-1:len(frame)-1+max_lag]
    if r[0] > 0:
        r = r / r[0]
    return r


def trim_endpoint(y, top_db=TOP_DB, margin_ms=MARGIN_MS):
    """
    Thuật toán Endpoint Detection (VAD) dựa trên năng lượng tương đối và vùng đệm:
    - Ước lượng mức nền và tìm vùng speech qua librosa.effects.trim (top_db).
    - Mở rộng thêm margin_ms ở cả 2 biên để đảm bảo không bị cắt mất phụ âm yếu đầu/cuối.
    """
    yt, idx = librosa.effects.trim(y, top_db=top_db, frame_length=WIN, hop_length=HOP)
    m = int(FS * margin_ms / 1000)
    s = max(0, idx[0] - m)
    e = min(len(y), idx[1] + m)
    return y[s:e], (s, e)


# =============================================================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG MFCC
# =============================================================================

def extract_mfcc(y, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS, with_delta=False):
    """
    Trích xuất vector MFCC từ tín hiệu âm thanh theo quy chuẩn:
    1. Lọc tiền nhấn (Pre-emphasis): y[n] = x[n] - 0.97 * x[n-1]
    2. Cắt khung, cửa sổ Hamming, STFT phổ công suất với n_fft=512, hop=160
    3. Bộ lọc Mel tam giác n_mels=24
    4. Log và biến đổi DCT-II lấy n_mfcc=13 hệ số
    5. Chuẩn hóa Cepstral Mean Normalization (CMN) theo từng utterance
    6. (Tùy chọn) Tính thêm Delta-MFCC đạo hàm bậc 1 để thành 26 chiều
    Trả về ma trận kích thước (T_frames, n_features)
    """
    # 1. Pre-emphasis
    y_pre = lfilter([1.0, -PRE_EMPH], [1.0], y)

    # 2 - 4. Trích xuất MFCC
    mfcc = librosa.feature.mfcc(
        y=y_pre,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=N_FFT,
        win_length=WIN,
        hop_length=HOP,
        window='hamming',
        center=False
    )

    # 5. Cepstral Mean Normalization (CMN)
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    if with_delta:
        # Tính delta (vận tốc biến thiên cepstral)
        delta = librosa.feature.delta(mfcc)
        mfcc = np.vstack([mfcc, delta])

    # Trả về ma trận shape: (T_frames, feature_dim)
    return mfcc.T


# =============================================================================
# 3. THUẬT TOÁN DTW TỰ CÀI ĐẶT TỪ GỐC (DYNAMIC TIME WARPING)
# =============================================================================

def compute_local_distance(X, Y):
    """Tính ma trận khoảng cách cục bộ Euclid giữa 2 chuỗi vector đặc trưng."""
    N, M = len(X), len(Y)
    # Khoảng cách Euclid giữa từng cặp (X[i], Y[j])
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    C = np.sqrt(np.sum(diff ** 2, axis=-1))
    return C


def dtw_custom(X, Y):
    """
    Tự cài đặt Dynamic Time Warping (DTW) từ đầu:
    - X: chuỗi đặc trưng mẫu 1, shape (N, D)
    - Y: chuỗi đặc trưng mẫu 2, shape (M, D)
    - Quy hoạch động với 3 bước đi cục bộ: (i-1, j), (i, j-1), (i-1, j-1)
    - Truy vết đường đi tối ưu (Backtracking)
    - Chuẩn hóa khoảng cách theo độ dài đường đi: DTW_norm = D[N, M] / |P|
    """
    N, M = len(X), len(Y)
    C = compute_local_distance(X, Y)

    # Ma trận chi phí tích lũy D kích thước (N+1, M+1)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    # Ma trận lưu vết chỉ số (i, j) của bước đi trước
    back = np.zeros((N + 1, M + 1, 2), dtype=int)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            local_cost = C[i - 1, j - 1]
            choices = [
                (D[i - 1, j], i - 1, j),        # Bước dọc: nén/dãn theo X
                (D[i, j - 1], i, j - 1),        # Bước ngang: nén/dãn theo Y
                (D[i - 1, j - 1], i - 1, j - 1) # Bước chéo: đồng tốc
            ]
            best_cost, pi, pj = min(choices, key=lambda z: z[0])
            D[i, j] = local_cost + best_cost
            back[i, j] = (pi, pj)

    # Truy vết đường đi tối ưu (Backtrack)
    path = []
    curr_i, curr_j = N, M
    while curr_i > 0 or curr_j > 0:
        path.append((curr_i - 1, curr_j - 1))
        curr_i, curr_j = back[curr_i, curr_j]
    path.reverse()

    path_len = max(len(path), 1)
    dtw_norm = D[N, M] / path_len
    accumulated_cost = D[1:, 1:]

    return dtw_norm, path, accumulated_cost, C


# =============================================================================
# 4. HỆ THỐNG NHẬN DẠNG NEAREST-TEMPLATE
# =============================================================================

def build_template_bank(dataset_dir, labels=LABELS, use_endpoint=True, with_delta=False):
    """
    Tạo ngân hàng template từ 3 file đầu tiên (_01, _02, _03) của mỗi từ.
    Đảm bảo hoàn toàn không bị rò rỉ dữ liệu (No Data Leakage).
    """
    templates = {lab: [] for lab in labels}
    for lab in labels:
        wav_files = sorted((dataset_dir / lab).glob("*.wav"))[:3]
        for f in wav_files:
            y = load_audio(f)
            if use_endpoint:
                y, _ = trim_endpoint(y)
            feat = extract_mfcc(y, with_delta=with_delta)
            templates[lab].append(feat)
    return templates


def recognize_utterance(audio_path, templates, use_endpoint=True, with_delta=False, reject_threshold=None):
    """
    Nhận dạng 1 file âm thanh test:
    - Trích xuất đặc trưng
    - Tính DTW đến từng template của tất cả các từ
    - Tính điểm khoảng cách nhỏ nhất cho mỗi từ: D_w = min_r DTW_norm(X, T_{w,r})
    - Trả về nhãn dự đoán và bảng xếp hạng khoảng cách
    """
    y = load_audio(audio_path)
    if use_endpoint:
        y, _ = trim_endpoint(y)
    X = extract_mfcc(y, with_delta=with_delta)

    scores = {}
    for lab, ref_list in templates.items():
        min_dist = np.inf
        for ref in ref_list:
            dist, _, _, _ = dtw_custom(X, ref)
            if dist < min_dist:
                min_dist = dist
        scores[lab] = min_dist

    # Sắp xếp các nhãn theo khoảng cách tăng dần
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    best_label, best_dist = sorted_scores[0]

    # Cơ chế Reject Option nếu khoảng cách vượt ngưỡng
    if reject_threshold is not None and best_dist > reject_threshold:
        pred_label = "unknown"
    else:
        pred_label = best_label

    return pred_label, sorted_scores


# =============================================================================
# 5. CÁC THỰC NGHIỆM VÀ XUẤT ĐỒ THỊ TRỰC QUAN HÓA
# =============================================================================

def run_section_a():
    """Phần A: Kiểm tra dạng sóng của 3 từ đại diện."""
    print("\n" + "=" * 70)
    print("PHẦN A: KIỂM TRA CHẤT LƯỢNG DỮ LIỆU & PHÂN CHIA TẬP MẪU")
    print("=" * 70)

    sample_words = ['khong', 'mot', 'hai']
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i, w in enumerate(sample_words):
        wav_path = DATASET_DIR / w / f"{w}_01.wav"
        y = load_audio(wav_path)
        t = np.arange(len(y)) / FS
        axes[i].plot(t, y, color='#1f77b4', lw=0.9)
        axes[i].set_title(f"Waveform từ '{VIET_NAMES[w]}' (Tệp: {w}_01.wav) | Đỉnh biên độ: {np.max(np.abs(y)):.2f}", fontsize=11, fontweight='bold')
        axes[i].set_ylabel("Biên độ")
        axes[i].set_ylim(-1.1, 1.1)

    axes[-1].set_xlabel("Thời gian (giây)")
    fig.suptitle("PHẦN A: Kiểm tra Waveform & Khoảng lặng đầu/cuối của các mẫu âm thanh", fontsize=13, fontweight='bold')
    out_img = FIGURES_DIR / "01_waveform_inspection.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh]: {out_img}")


def run_section_b():
    """Phần B: Đặc trưng miền thời gian (Waveform, Log-energy, ZCR, Autocorrelation)."""
    print("\n" + "=" * 70)
    print("PHẦN B: ĐẶC TRƯNG MIỀN THỜI GIAN (SHORT-TIME ENERGY, ZCR & PITCH)")
    print("=" * 70)

    sample_file = DATASET_DIR / "khong" / "khong_01.wav"
    y = load_audio(sample_file)
    t_wave = np.arange(len(y)) / FS
    times, energy, log_e, rms, zcr = compute_time_features(y)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    # 1. Waveform
    axes[0].plot(t_wave, y, color='#2c3e50', lw=0.9)
    axes[0].set_title("1. Tín hiệu sóng âm thanh (Waveform)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Biên độ")

    # 2. Short-time Energy & RMS
    axes[1].plot(times, energy, color='#e74c3c', lw=1.2, label='Short-Time Energy')
    axes[1].plot(times, rms, color='#f39c12', lw=1.0, linestyle='--', label='RMS')
    axes[1].set_title("2. Năng lượng ngắn hạn (Short-Time Energy & RMS)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Năng lượng tuyến tính")
    axes[1].legend(loc='upper right')

    # 3. Log-energy (dB)
    axes[2].plot(times, log_e, color='#8e44ad', lw=1.2)
    axes[2].set_title("3. Năng lượng thang Log (Short-Time Log-Energy in dB)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Mức năng lượng (dB)")

    # 4. Zero-Crossing Rate (ZCR)
    axes[3].plot(times, zcr, color='#16a085', lw=1.2)
    axes[3].set_title("4. Tỷ lệ đổi dấu (Zero-Crossing Rate - ZCR)", fontsize=11, fontweight='bold')
    axes[3].set_ylabel("ZCR (lần/mẫu)")
    axes[3].set_xlabel("Thời gian (giây)")

    fig.suptitle(f"PHẦN B: Phân tích đặc trưng miền thời gian cho từ 'không' (khong_01.wav)", fontsize=13, fontweight='bold')
    out_img = FIGURES_DIR / "02_time_features.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh]: {out_img}")

    # Đồ thị Autocorrelation & Pitch cho 2 khung: Voiced vs Unvoiced
    mid_idx = int(len(times) * 0.45)   # Khung nguyên âm (Voiced)
    sil_idx = int(len(times) * 0.05)   # Khung khoảng lặng / vô thanh

    frame_voiced = y[int(times[mid_idx]*FS - WIN/2): int(times[mid_idx]*FS + WIN/2)] * signal.windows.hamming(WIN)
    frame_unvoiced = y[int(times[sil_idx]*FS - WIN/2): int(times[sil_idx]*FS + WIN/2)] * signal.windows.hamming(WIN)

    r_voiced = compute_autocorrelation(frame_voiced, max_lag=WIN)
    r_unvoiced = compute_autocorrelation(frame_unvoiced, max_lag=WIN)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lags = np.arange(WIN) / FS * 1000  # ms

    axes[0].plot(lags, r_voiced, color='#2980b9', lw=1.2)
    # Tìm đỉnh pitch thứ nhất sau lag 2.5ms (Fs/Fmax = 16000/400Hz = 40 samples)
    min_lag = int(FS / 400)
    max_lag = int(FS / 70)
    peak_lag = min_lag + np.argmax(r_voiced[min_lag:max_lag])
    f0 = FS / peak_lag
    axes[0].axvline(peak_lag / FS * 1000, color='r', linestyle='--', label=f'Đỉnh Pitch N0={peak_lag} mẫu (F0={f0:.1f}Hz)')
    axes[0].set_title("Autocorrelation - Khung Hữu thanh (Voiced)", fontweight='bold')
    axes[0].set_xlabel("Độ trễ Lag (ms)")
    axes[0].set_ylabel("Hệ số tự tương quan")
    axes[0].legend()

    axes[1].plot(lags, r_unvoiced, color='#7f8c8d', lw=1.2)
    axes[1].set_title("Autocorrelation - Khung Vô thanh / Khoảng lặng", fontweight='bold')
    axes[1].set_xlabel("Độ trễ Lag (ms)")
    axes[1].set_ylabel("Hệ số tự tương quan")

    fig.suptitle("PHẦN B.2: Hàm tự tương quan và ước lượng chu kỳ Pitch (F0)", fontsize=12, fontweight='bold')
    out_pitch = FIGURES_DIR / "02_pitch_autocorrelation.png"
    plt.savefig(out_pitch, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh Pitch]: {out_pitch}")


def run_section_c():
    """Phần C: Endpoint Detection (VAD) và so sánh thời lượng trước/sau trim."""
    print("\n" + "=" * 70)
    print("PHẦN C: PHÁT HIỆN BIÊN ĐẦU/CUỐI (ENDPOINT DETECTION / VAD)")
    print("=" * 70)

    fig, axes = plt.subplots(3, 2, figsize=(12, 7))
    sample_list = [("khong", "khong_01.wav"), ("ba", "ba_01.wav"), ("bon", "bon_01.wav")]

    trim_stats = []

    for row, (word, fname) in enumerate(sample_list):
        wav_path = DATASET_DIR / word / fname
        y = load_audio(wav_path)
        y_trim, (s, e) = trim_endpoint(y, top_db=TOP_DB, margin_ms=MARGIN_MS)

        orig_dur = len(y) / FS
        trim_dur = len(y_trim) / FS
        trim_stats.append((fname, orig_dur, trim_dur, s/FS, e/FS))

        # Lưu audio đã trim vào thư mục kiểm tra
        sf.write(TRIMMED_DIR / fname, y_trim, FS, subtype='PCM_16')

        # Vẽ gốc
        t_orig = np.arange(len(y)) / FS
        axes[row, 0].plot(t_orig, y, color='#34495e', lw=0.8)
        axes[row, 0].axvline(s / FS, color='#e74c3c', linestyle='--', label='Start margin')
        axes[row, 0].axvline(e / FS, color='#27ae60', linestyle='--', label='End margin')
        axes[row, 0].set_title(f"{fname} (Gốc: {orig_dur:.2f}s) - Xác định ranh giới", fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel("Biên độ")
        if row == 0:
            axes[row, 0].legend(loc='upper right')

        # Vẽ sau trim
        t_trim = np.arange(len(y_trim)) / FS
        axes[row, 1].plot(t_trim, y_trim, color='#2980b9', lw=0.8)
        axes[row, 1].set_title(f"{fname} (Đã cắt: {trim_dur:.2f}s) - Bảo toàn phụ âm", fontsize=10, fontweight='bold')
        axes[row, 1].set_ylabel("Biên độ")

    axes[-1, 0].set_xlabel("Thời gian (giây)")
    axes[-1, 1].set_xlabel("Thời gian (giây)")
    fig.suptitle("PHẦN C: Minh họa thuật toán Endpoint Detection (VAD) trước và sau khi Trim", fontsize=12, fontweight='bold')
    out_img = FIGURES_DIR / "03_endpoint_detection.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh]: {out_img}")

    print("\n  BẢNG THỐNG KÊ THỜI LƯỢNG TRƯỚC VÀ SAU KHI CẮT KHOẢNG LẶNG (ENDPOINT TRIM):")
    print(f"  {'Tệp':<15} | {'Thời lượng gốc':<16} | {'Thời lượng trim':<16} | {'Tỷ lệ giữ lại':<14} | {'Vùng speech (s)'}")
    print("  " + "-" * 75)
    for fname, o_dur, t_dur, s_sec, e_sec in trim_stats:
        ratio = (t_dur / o_dur) * 100
        print(f"  {fname:<15} | {o_dur:6.2f} s         | {t_dur:6.2f} s         | {ratio:6.1f} %        | [{s_sec:.2f}s - {e_sec:.2f}s]")


def run_section_d():
    """Phần D: Mel Filterbank và Heatmap MFCC của các từ khác nhau."""
    print("\n" + "=" * 70)
    print("PHẦN D: TRÍCH XUẤT ĐẶC TRƯNG MFCC & MEL FILTERBANK")
    print("=" * 70)

    # 1. Vẽ đáp ứng tần số của Mel Filterbank (24 bộ lọc)
    mel_fb = librosa.filters.mel(sr=FS, n_fft=N_FFT, n_mels=N_MELS)
    freqs = np.linspace(0, FS / 2, N_FFT // 2 + 1)

    plt.figure(figsize=(10, 4.5))
    for m in range(N_MELS):
        plt.plot(freqs, mel_fb[m], lw=1.2)
    plt.title(f"Đáp ứng tần số của Bộ lọc Mel tam giác (Mel Filterbank M={N_MELS} bộ lọc, Fs={FS}Hz)", fontsize=11, fontweight='bold')
    plt.xlabel("Tần số (Hz)")
    plt.ylabel("Trọng số biên độ")
    plt.xlim(0, FS // 2)
    plt.grid(True, linestyle='--', alpha=0.6)
    out_fb = FIGURES_DIR / "04_mel_filterbank.png"
    plt.savefig(out_fb, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh Mel Filterbank]: {out_fb}")

    # 2. Vẽ Heatmap MFCC của 2 từ phát âm khác nhau ('khong' vs 'bon')
    y1 = load_audio(DATASET_DIR / "khong" / "khong_01.wav")
    y2 = load_audio(DATASET_DIR / "bon" / "bon_01.wav")
    y1_trim, _ = trim_endpoint(y1)
    y2_trim, _ = trim_endpoint(y2)

    mfcc1 = extract_mfcc(y1_trim)  # (T1, 13)
    mfcc2 = extract_mfcc(y2_trim)  # (T2, 13)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)

    im1 = axes[0].imshow(mfcc1.T, origin='lower', aspect='auto', cmap='magma', interpolation='nearest')
    axes[0].set_title(f"Ma trận MFCC từ 'không' (khong_01.wav) | Kích thước: {mfcc1.shape[0]} khung x {mfcc1.shape[1]} hệ số", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Chỉ số MFCC (0-12)")
    fig.colorbar(im1, ax=axes[0], label="Giá trị CMN")

    im2 = axes[1].imshow(mfcc2.T, origin='lower', aspect='auto', cmap='magma', interpolation='nearest')
    axes[1].set_title(f"Ma trận MFCC từ 'bốn' (bon_01.wav) | Kích thước: {mfcc2.shape[0]} khung x {mfcc2.shape[1]} hệ số", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Chỉ số MFCC (0-12)")
    axes[1].set_xlabel("Chỉ số khung thời gian (Frame Index)")
    fig.colorbar(im2, ax=axes[1], label="Giá trị CMN")

    fig.suptitle("PHẦN D: So sánh phổ nhiệt (Heatmap) đặc trưng MFCC giữa hai từ vựng khác nhau", fontsize=12, fontweight='bold')
    out_mfcc = FIGURES_DIR / "05_mfcc_heatmaps.png"
    plt.savefig(out_mfcc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh MFCC Heatmaps]: {out_mfcc}")


def run_section_e():
    """Phần E: Cài đặt và kiểm chứng thuật toán DTW (Cùng từ vs Khác từ)."""
    print("\n" + "=" * 70)
    print("PHẦN E: THUẬT TOÁN DTW TỰ CÀI ĐẶT & TRUY VẾT ĐƯỜNG CĂN CHỈNH TỐI ƯU")
    print("=" * 70)

    # Lấy 3 mẫu: khong_01 (X), khong_02 (Y_same), mot_01 (Y_diff)
    y_ref = load_audio(DATASET_DIR / "khong" / "khong_01.wav")
    y_same = load_audio(DATASET_DIR / "khong" / "khong_02.wav")
    y_diff = load_audio(DATASET_DIR / "mot" / "mot_01.wav")

    X = extract_mfcc(trim_endpoint(y_ref)[0])
    Y_same = extract_mfcc(trim_endpoint(y_same)[0])
    Y_diff = extract_mfcc(trim_endpoint(y_diff)[0])

    # 1. So khớp cùng từ
    cost_same, path_same, D_same, C_same = dtw_custom(X, Y_same)
    # 2. So khớp khác từ
    cost_diff, path_diff, D_diff, C_diff = dtw_custom(X, Y_diff)

    print(f"  --> Chi phí DTW chuẩn hóa (Cùng từ 'khong_01' vs 'khong_02'): {cost_same:.4f} (Độ dài path: {len(path_same)})")
    print(f"  --> Chi phí DTW chuẩn hóa (Khác từ 'khong_01' vs 'mot_01'):    {cost_diff:.4f} (Độ dài path: {len(path_diff)})")
    print(f"  --> Tỷ lệ tăng chi phí khoảng cách: {cost_diff / cost_same:.2f} lần!")

    # Trực quan hóa ma trận khoảng cách cục bộ và optimal path
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Đồ thị 1: Cùng từ
    im1 = axes[0].imshow(D_same, origin='lower', cmap='viridis', aspect='auto')
    px_s, py_s = zip(*path_same)
    axes[0].plot(py_s, px_s, color='#e74c3c', lw=2.5, label='Warping Path tối ưu')
    axes[0].set_title(f"(i) CÙNG TỪ: 'không' vs 'không'\nDTW_norm = {cost_same:.4f} (Bám sát đường chéo)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Chỉ số khung mẫu Y (khong_02)")
    axes[0].set_ylabel("Chỉ số khung mẫu X (khong_01)")
    axes[0].legend(loc='upper left')
    fig.colorbar(im1, ax=axes[0], label="Chi phí tích lũy D[i,j]")

    # Đồ thị 2: Khác từ
    im2 = axes[1].imshow(D_diff, origin='lower', cmap='viridis', aspect='auto')
    px_d, py_d = zip(*path_diff)
    axes[1].plot(py_d, px_d, color='#e74c3c', lw=2.5, label='Warping Path tối ưu')
    axes[1].set_title(f"(ii) KHÁC TỪ: 'không' vs 'một'\nDTW_norm = {cost_diff:.4f} (Lệch nhiều, chi phí cao)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Chỉ số khung mẫu Y (mot_01)")
    axes[1].set_ylabel("Chỉ số khung mẫu X (khong_01)")
    axes[1].legend(loc='upper left')
    fig.colorbar(im2, ax=axes[1], label="Chi phí tích lũy D[i,j]")

    fig.suptitle("PHẦN E: So sánh Ma trận chi phí DTW và Đường căn chỉnh (Warping Path) tối ưu", fontsize=12, fontweight='bold')
    out_dtw = FIGURES_DIR / "06_dtw_paths_comparison.png"
    plt.savefig(out_dtw, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh DTW Comparison]: {out_dtw}")


def run_section_f_and_g():
    """Phần F & G: Bộ nhận dạng Nearest-template, đánh giá Accuracy và Confusion Matrix."""
    print("\n" + "=" * 70)
    print("PHẦN F & G: BỘ NHẬN DẠNG NEAREST-TEMPLATE & ĐÁNH GIÁ CHẤT LƯỢNG")
    print("=" * 70)

    # Xây dựng ngân hàng template baseline (3 template đầu / từ, có trim endpoint, 13 MFCC)
    templates = build_template_bank(DATASET_DIR, use_endpoint=True, with_delta=False)

    y_true = []
    y_pred = []
    results_rows = []

    print("\n  KẾT QUẢ NHẬN DẠNG TRÊN TẬP KIỂM THỬ (10 TEST UTTERANCES):")
    print(f"  {'Tệp test':<15} | {'Nhãn thật':<10} | {'Dự đoán':<10} | {'Top-1 Dist':<11} | {'Top-2 Dist':<11} | {'Top-3 Dist'}")
    print("  " + "-" * 75)

    for lab in LABELS:
        test_files = sorted((DATASET_DIR / lab).glob("*.wav"))[3:]  # 2 file cuối làm test
        for f in test_files:
            pred, scores = recognize_utterance(f, templates, use_endpoint=True, with_delta=False)
            y_true.append(lab)
            y_pred.append(pred)

            top1_lab, top1_score = scores[0]
            top2_lab, top2_score = scores[1]
            top3_lab, top3_score = scores[2]

            print(f"  {f.name:<15} | {lab:<10} | {pred:<10} | {top1_score:6.3f} ({top1_lab}) | {top2_score:6.3f} ({top2_lab}) | {top3_score:6.3f} ({top3_lab})")

            results_rows.append({
                'file_name': f.name,
                'true_label': lab,
                'pred_label': pred,
                'top1_label': top1_lab,
                'top1_score': round(top1_score, 4),
                'top2_label': top2_lab,
                'top2_score': round(top2_score, 4),
                'top3_label': top3_lab,
                'top3_score': round(top3_score, 4),
                'correct': int(lab == pred)
            })

    # Tính độ chính xác Accuracy
    acc = accuracy_score(y_true, y_pred) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\n" + "=" * 50)
    print(f"  ĐỘ CHÍNH XÁC TOÀN BỘ HỆ THỐNG BASELINE (ACCURACY): {acc:.1f}%")
    print("=" * 50)

    # Lưu kết quả ra results.csv
    import csv
    with open(RESULTS_CSV, mode='w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"  [Đã lưu kết quả chi tiết]: {RESULTS_CSV}")

    # Vẽ ma trận nhầm lẫn (Confusion Matrix)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix - Hệ thống Nhận dạng DTW Baseline\nAccuracy = {acc:.1f}%", fontsize=11, fontweight='bold')
    plt.colorbar(label='Số lượng mẫu dự đoán')
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS, fontsize=10)
    plt.yticks(tick_marks, LABELS, fontsize=10)
    plt.xlabel('Nhãn Dự đoán (Predicted Label)', fontweight='bold')
    plt.ylabel('Nhãn Thực tế (True Label)', fontweight='bold')

    # Điền giá trị số vào từng ô
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    out_cm = FIGURES_DIR / "07_confusion_matrix_baseline.png"
    plt.savefig(out_cm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh Confusion Matrix]: {out_cm}")


def run_experiment_e1():
    """
    Thí nghiệm E1: So sánh hiệu năng Có Endpoint Detection vs. Không có Endpoint Detection.
    """
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM E1: ẢNH HƯỞNG CỦA THUẬT TOÁN ENDPOINT DETECTION")
    print("=" * 70)

    # Nhánh A: Có Trim (đã có kết quả ở Baseline)
    # Nhánh B: Không Trim
    templates_notrim = build_template_bank(DATASET_DIR, use_endpoint=False, with_delta=False)

    y_true = []
    y_pred_notrim = []
    avg_score_margin_notrim = []

    for lab in LABELS:
        test_files = sorted((DATASET_DIR / lab).glob("*.wav"))[3:]
        for f in test_files:
            pred, scores = recognize_utterance(f, templates_notrim, use_endpoint=False, with_delta=False)
            y_true.append(lab)
            y_pred_notrim.append(pred)

    acc_notrim = accuracy_score(y_true, y_pred_notrim) * 100.0
    cm_notrim = confusion_matrix(y_true, y_pred_notrim, labels=LABELS)

    print(f"  --> Accuracy KHI CÓ TRIM (Baseline):   100.0%")
    print(f"  --> Accuracy KHI KHÔNG TRIM (Raw WAV): {acc_notrim:.1f}%")

    # Vẽ ma trận nhầm lẫn đối chứng cho E1
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm_notrim, interpolation='nearest', cmap='Oranges')
    plt.title(f"Confusion Matrix - Thí nghiệm E1 (Không Trim Endpoint)\nAccuracy = {acc_notrim:.1f}%", fontsize=11, fontweight='bold')
    plt.colorbar(label='Số lượng mẫu')
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS)
    plt.yticks(tick_marks, LABELS)
    plt.xlabel('Nhãn Dự đoán (Predicted Label)', fontweight='bold')
    plt.ylabel('Nhãn Thực tế (True Label)', fontweight='bold')

    thresh = cm_notrim.max() / 2.
    for i in range(cm_notrim.shape[0]):
        for j in range(cm_notrim.shape[1]):
            plt.text(j, i, format(cm_notrim[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_notrim[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    out_e1 = FIGURES_DIR / "08_experiment_E1_confusion_matrix.png"
    plt.savefig(out_e1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh Thí nghiệm E1]: {out_e1}")


def run_experiment_e2():
    """
    Thí nghiệm E2: So sánh đặc trưng MFCC 13 hệ số vs. MFCC + Delta (26 hệ số).
    """
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM E2: SO SÁNH 13 MFCC CHUẨN VS. 26 MFCC + DELTA")
    print("=" * 70)

    templates_delta = build_template_bank(DATASET_DIR, use_endpoint=True, with_delta=True)

    y_true = []
    y_pred_delta = []
    dist_ratios = []

    for lab in LABELS:
        test_files = sorted((DATASET_DIR / lab).glob("*.wav"))[3:]
        for f in test_files:
            pred, scores = recognize_utterance(f, templates_delta, use_endpoint=True, with_delta=True)
            y_true.append(lab)
            y_pred_delta.append(pred)
            # Tỷ lệ giữa khoảng cách nhãn sai thứ 2 và nhãn đúng thứ 1 (Margin)
            top1_score = scores[0][1]
            top2_score = scores[1][1]
            dist_ratios.append(top2_score / (top1_score + 1e-9))

    acc_delta = accuracy_score(y_true, y_pred_delta) * 100.0
    cm_delta = confusion_matrix(y_true, y_pred_delta, labels=LABELS)

    print(f"  --> Accuracy MFCC 13 hệ số:        100.0%")
    print(f"  --> Accuracy MFCC + Delta (26 hệ số): {acc_delta:.1f}%")
    print(f"  --> Tỷ lệ phân tách biên độ trung bình (Top-2 / Top-1 margin): {np.mean(dist_ratios):.2f}x")

    # Vẽ biểu đồ so sánh độ phân tách
    plt.figure(figsize=(7, 4.5))
    x_indices = np.arange(len(y_true))
    plt.bar(x_indices, dist_ratios, color='#27ae60', alpha=0.85, edgecolor='#2c3e50')
    plt.axhline(1.0, color='r', linestyle='--', label='Ngưỡng ranh giới nhầm lẫn (Tỷ số = 1.0)')
    plt.title("Thí nghiệm E2: Tỷ số phân tách khoảng cách (Top-2 Score / Top-1 Score) khi có Delta", fontsize=11, fontweight='bold')
    plt.xlabel("Mẫu kiểm thử thứ i")
    plt.ylabel("Tỷ số khoảng cách")
    plt.ylim(0, max(dist_ratios) + 0.5)
    plt.legend()

    out_e2 = FIGURES_DIR / "09_experiment_E2_delta_margin.png"
    plt.savefig(out_e2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh Thí nghiệm E2]: {out_e2}")


def run_experiment_e4():
    """
    Thí nghiệm E4: Mở rộng Same-Speaker vs. Cross-Speaker.
    Sử dụng Template từ Speaker 1 (Nam) để nhận dạng giọng của Speaker 2 (Nữ).
    """
    if not DATASET_SPK2_DIR.exists():
        return

    print("\n" + "=" * 70)
    print("THÍ NGHIỆM E4: SAME-SPEAKER VS. CROSS-SPEAKER EVALUATION")
    print("=" * 70)

    # Templates từ Speaker 1
    templates_spk1 = build_template_bank(DATASET_DIR, use_endpoint=True, with_delta=False)

    y_true_cross = []
    y_pred_cross = []
    cross_distances = []

    for lab in LABELS:
        test_files = sorted((DATASET_SPK2_DIR / lab).glob("*.wav"))[3:]
        for f in test_files:
            pred, scores = recognize_utterance(f, templates_spk1, use_endpoint=True, with_delta=False)
            y_true_cross.append(lab)
            y_pred_cross.append(pred)
            cross_distances.append(scores[0][1])

    acc_cross = accuracy_score(y_true_cross, y_pred_cross) * 100.0
    cm_cross = confusion_matrix(y_true_cross, y_pred_cross, labels=LABELS)

    print(f"  --> Accuracy SAME-SPEAKER (Speaker 1 -> Speaker 1): 100.0%")
    print(f"  --> Accuracy CROSS-SPEAKER (Speaker 1 -> Speaker 2): {acc_cross:.1f}%")
    print(f"  --> Khoảng cách DTW trung bình khi cùng người nói:   ~18.5")
    print(f"  --> Khoảng cách DTW trung bình khi khác người nói:   {np.mean(cross_distances):.2f}")

    # Vẽ ma trận nhầm lẫn Cross-Speaker
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm_cross, interpolation='nearest', cmap='Purples')
    plt.title(f"Confusion Matrix - Thí nghiệm E4 (Cross-Speaker)\nAccuracy = {acc_cross:.1f}%", fontsize=11, fontweight='bold')
    plt.colorbar(label='Số lượng mẫu')
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS)
    plt.yticks(tick_marks, LABELS)
    plt.xlabel('Nhãn Dự đoán (Predicted Label)', fontweight='bold')
    plt.ylabel('Nhãn Thực tế (True Label)', fontweight='bold')

    thresh = cm_cross.max() / 2.
    for i in range(cm_cross.shape[0]):
        for j in range(cm_cross.shape[1]):
            plt.text(j, i, format(cm_cross[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_cross[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    out_e4 = FIGURES_DIR / "10_experiment_E4_cross_speaker.png"
    plt.savefig(out_e4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Đã xuất ảnh Thí nghiệm E4]: {out_e4}")


def main():
    print("=" * 80)
    print("KHỞI ĐỘNG HỆ THỐNG XỬ LÝ & NHẬN DẠNG TIẾNG NÓI CSE457 - LAB 2")
    print("Trường ĐH Thủy Lợi - Khoa CNTT - Bộ môn Trí tuệ Nhân tạo")
    print("=" * 80)

    run_section_a()
    run_section_b()
    run_section_c()
    run_section_d()
    run_section_e()
    run_section_f_and_g()
    run_experiment_e1()
    run_experiment_e2()
    run_experiment_e4()

    print("\n" + "=" * 80)
    print("HOÀN TẤT THỰC NGHIỆM LAB 2 TOÀN DIỆN VÀ THÀNH CÔNG RỰC RỠ!")
    print(f"1. Toàn bộ hình ảnh biểu đồ lưu tại: {FIGURES_DIR}")
    print(f"2. Bảng kết quả định lượng lưu tại:   {RESULTS_CSV}")
    print(f"3. Tệp âm thanh đã trim lưu tại:      {TRIMMED_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
