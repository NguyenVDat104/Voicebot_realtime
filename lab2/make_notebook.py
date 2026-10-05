"""
Script tạo Notebook Lab02_Analysis.ipynb chuẩn định dạng Jupyter Notebook v4
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 2\n",
            "## ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG TỪ ĐƠN BẰNG DTW\n",
            "**Trường Đại học Thủy Lợi - Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo**\n",
            "\n",
            "---\n",
            "### Thông tin sinh viên\n",
            "- **Họ và tên:** Nguyễn Văn Đạt\n",
            "- **Mã sinh viên:** 215106xxxx\n",
            "- **Lớp:** CSE457\n",
            "- **Học phần:** Xử lý âm thanh và tiếng nói\n",
            "\n",
            "---\n",
            "### Mục tiêu thực hành\n",
            "1. Phân tích tín hiệu ngắn hạn: Phân đoạn khung (Framing), cửa sổ hóa (Hamming), Short-time Energy, RMS, ZCR và Autocorrelation (Pitch).\n",
            "2. Xây dựng thuật toán Endpoint Detection (VAD) để cắt bỏ khoảng lặng thừa ở đầu và cuối từ.\n",
            "3. Trích xuất đặc trưng MFCC (Mel-Frequency Cepstral Coefficients) kết hợp chuẩn hóa CMN.\n",
            "4. **Tự cài đặt thuật toán DTW (Dynamic Time Warping)** bằng quy hoạch động và truy vết đường đi tối ưu.\n",
            "5. Xây dựng bộ nhận dạng Nearest-Template cho 5 từ tiếng Việt (`không`, `một`, `hai`, `ba`, `bốn`).\n",
            "6. Thực hiện các thí nghiệm đối chứng: E1 (Endpoint Trim), E2 (Delta-MFCC) và E4 (Cross-Speaker)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Khởi tạo môi trường và nạp các thư viện cần thiết\n",
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import scipy.signal as signal\n",
            "from scipy.signal import lfilter\n",
            "import soundfile as sf\n",
            "import librosa\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.metrics import accuracy_score, confusion_matrix\n",
            "\n",
            "# Cấu hình hiển thị đồ thị\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
            "plt.rcParams['figure.autolayout'] = True\n",
            "\n",
            "# Hằng số hệ thống chuẩn theo đề cương Lab 2\n",
            "FS = 16000\n",
            "FRAME_MS = 25\n",
            "HOP_MS = 10\n",
            "WIN = int(FS * FRAME_MS / 1000)  # 400 mẫu\n",
            "HOP = int(FS * HOP_MS / 1000)    # 160 mẫu\n",
            "N_FFT = 512\n",
            "N_MELS = 24\n",
            "N_MFCC = 13\n",
            "PRE_EMPH = 0.97\n",
            "TOP_DB = 32\n",
            "MARGIN_MS = 50\n",
            "\n",
            "LABELS = ['khong', 'mot', 'hai', 'ba', 'bon']\n",
            "VIET_MAP = {'khong': 'không', 'mot': 'một', 'hai': 'hai', 'ba': 'ba', 'bon': 'bốn'}\n",
            "DATASET_DIR = Path('dataset')\n",
            "DATASET_SPK2_DIR = Path('dataset_speaker2')\n",
            "print('Cấu hình tham số hoàn tất. Sẵn sàng thực thi!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phần A: Dữ liệu và Kiểm tra Chất lượng Âm thanh\n",
            "- Bộ dữ liệu 5 từ tiếng Việt: `không`, `một`, `hai`, `ba`, `bốn`.\n",
            "- Mỗi từ gồm 5 file phát âm tách rời ở định dạng WAV mono 16kHz PCM.\n",
            "- Phân chia: 3 file đầu (`_01`, `_02`, `_03`) làm Template (Train), 2 file sau (`_04`, `_05`) làm Test độc lập."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def load_audio(path, target_fs=FS):\n",
            "    y, sr = librosa.load(path, sr=target_fs, mono=True)\n",
            "    max_val = np.max(np.abs(y))\n",
            "    if max_val > 1e-9:\n",
            "        y = y / max_val\n",
            "    return y\n",
            "\n",
            "# Vẽ dạng sóng waveform của 3 từ kiểm tra clipping và khoảng lặng đầu/cuối\n",
            "sample_words = ['khong', 'mot', 'hai']\n",
            "fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)\n",
            "for i, w in enumerate(sample_words):\n",
            "    y = load_audio(DATASET_DIR / w / f\"{w}_01.wav\")\n",
            "    t = np.arange(len(y)) / FS\n",
            "    axes[i].plot(t, y, color='#1f77b4', lw=0.9)\n",
            "    axes[i].set_title(f\"Waveform từ '{VIET_MAP[w]}' ({w}_01.wav) | Peak: {np.max(np.abs(y)):.2f}\", fontweight='bold')\n",
            "    axes[i].set_ylabel(\"Biên độ\")\n",
            "axes[-1].set_xlabel(\"Thời gian (giây)\")\n",
            "plt.suptitle(\"Phần A: Kiểm tra dạng sóng và khoảng lặng (Silence) của các mẫu âm thanh\", fontsize=12, fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phần B: Phân tích Đặc trưng Miền Thời gian\n",
            "- Short-time Energy: $E_r = \\sum_n x_r^2[n]$\n",
            "- Zero-Crossing Rate (ZCR): $Z_r = \\frac{1}{2L} \\sum_m |\\text{sgn}(x[m]) - \\text{sgn}(x[m-1])|$\n",
            "- So sánh sự khác nhau rõ rệt giữa: **Silence** (năng lượng cực thấp), **Voiced** (năng lượng cao, ZCR thấp) và **Unvoiced** (năng lượng thấp, ZCR cao)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def compute_time_features(y, frame_len=WIN, hop_len=HOP):\n",
            "    num_frames = 1 + int((len(y) - frame_len) / hop_len)\n",
            "    energy, rms, zcr = np.zeros(num_frames), np.zeros(num_frames), np.zeros(num_frames)\n",
            "    times = (np.arange(num_frames) * hop_len + frame_len / 2) / FS\n",
            "    window = signal.windows.hamming(frame_len)\n",
            "    for i in range(num_frames):\n",
            "        start = i * hop_len\n",
            "        frame_win = y[start:start+frame_len] * window\n",
            "        energy[i] = np.sum(frame_win ** 2)\n",
            "        rms[i] = np.sqrt(np.mean(frame_win ** 2))\n",
            "        signs = np.sign(y[start:start+frame_len])\n",
            "        signs[signs == 0] = 1\n",
            "        zcr[i] = np.sum(np.abs(signs[1:] - signs[:-1])) / (2.0 * frame_len)\n",
            "    log_e = 10 * np.log10(energy + 1e-12)\n",
            "    return times, energy, log_e, rms, zcr\n",
            "\n",
            "y = load_audio(DATASET_DIR / 'khong' / 'khong_01.wav')\n",
            "t_wave = np.arange(len(y)) / FS\n",
            "times, energy, log_e, rms, zcr = compute_time_features(y)\n",
            "\n",
            "fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)\n",
            "axes[0].plot(t_wave, y, color='#2c3e50', lw=0.9)\n",
            "axes[0].set_title(\"1. Waveform\", fontweight='bold'); axes[0].set_ylabel(\"Biên độ\")\n",
            "axes[1].plot(times, log_e, color='#e74c3c', lw=1.2)\n",
            "axes[1].set_title(\"2. Short-Time Log-Energy (dB)\", fontweight='bold'); axes[1].set_ylabel(\"Mức (dB)\")\n",
            "axes[2].plot(times, zcr, color='#16a085', lw=1.2)\n",
            "axes[2].set_title(\"3. Zero-Crossing Rate (ZCR)\", fontweight='bold'); axes[2].set_ylabel(\"Tỷ lệ\"); axes[2].set_xlabel(\"Thời gian (s)\")\n",
            "plt.suptitle(\"Phần B: Đặc trưng miền thời gian của từ 'không'\", fontsize=12, fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phần C: Thuật toán Phát hiện Điểm Đầu/Cuối (Endpoint Detection / VAD)\n",
            "- Sử dụng năng lượng tương đối (`top_db=32`) kết hợp vùng đệm an toàn (`margin_ms=50ms`).\n",
            "- Vùng đệm 50ms giúp bảo toàn hoàn toàn các phụ âm đầu (/k/, /b/, /m/) và phụ âm cuối."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def trim_endpoint(y, top_db=TOP_DB, margin_ms=MARGIN_MS):\n",
            "    yt, idx = librosa.effects.trim(y, top_db=top_db, frame_length=WIN, hop_length=HOP)\n",
            "    m = int(FS * margin_ms / 1000)\n",
            "    s = max(0, idx[0] - m)\n",
            "    e = min(len(y), idx[1] + m)\n",
            "    return y[s:e], (s, e)\n",
            "\n",
            "# Kiểm tra cắt khoảng lặng trên 3 từ có phụ âm yếu\n",
            "sample_list = [('khong', 'khong_01.wav'), ('ba', 'ba_01.wav'), ('bon', 'bon_01.wav')]\n",
            "print(f\"{'Tệp':<15} | {'Thời lượng gốc':<16} | {'Thời lượng trim':<16} | {'Tỷ lệ giữ lại'}\")\n",
            "print('-' * 65)\n",
            "for word, fname in sample_list:\n",
            "    y = load_audio(DATASET_DIR / word / fname)\n",
            "    y_trim, (s, e) = trim_endpoint(y)\n",
            "    o_dur = len(y)/FS; t_dur = len(y_trim)/FS\n",
            "    print(f\"{fname:<15} | {o_dur:6.2f} s         | {t_dur:6.2f} s         | {(t_dur/o_dur)*100:6.1f}%\")\n",
            "\n",
            "# Vẽ đối chiếu trước và sau trim\n",
            "y = load_audio(DATASET_DIR / 'khong' / 'khong_01.wav')\n",
            "y_t, (s, e) = trim_endpoint(y)\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.plot(np.arange(len(y))/FS, y, color='#7f8c8d', alpha=0.7, label='Tín hiệu gốc có silence')\n",
            "plt.axvline(s/FS, color='r', linestyle='--', label='Start margin')\n",
            "plt.axvline(e/FS, color='g', linestyle='--', label='End margin')\n",
            "plt.plot(np.arange(s, e)/FS, y[s:e], color='#2980b9', lw=1.2, label='Đoạn speech được giữ lại')\n",
            "plt.title(\"Phần C: Minh họa thuật toán Endpoint Detection (VAD)\", fontweight='bold')\n",
            "plt.xlabel(\"Thời gian (giây)\"); plt.ylabel(\"Biên độ\"); plt.legend()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phần D: Trích xuất Đặc trưng MFCC & Mel Filterbank\n",
            "- Pre-emphasis: $\\alpha = 0.97$\n",
            "- Bộ lọc Mel: $M = 24$ bộ lọc tam giác\n",
            "- MFCC: 13 hệ số đầu tiên\n",
            "- Chuẩn hóa CMN (Cepstral Mean Normalization) theo từng utterance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def extract_mfcc(y, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS, with_delta=False):\n",
            "    y_pre = lfilter([1.0, -PRE_EMPH], [1.0], y)\n",
            "    mfcc = librosa.feature.mfcc(\n",
            "        y=y_pre, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=N_FFT,\n",
            "        win_length=WIN, hop_length=HOP, window='hamming', center=False\n",
            "    )\n",
            "    # Chuẩn hóa CMN\n",
            "    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)\n",
            "    if with_delta:\n",
            "        delta = librosa.feature.delta(mfcc)\n",
            "        mfcc = np.vstack([mfcc, delta])\n",
            "    return mfcc.T  # (T_frames, n_features)\n",
            "\n",
            "# Vẽ ma trận MFCC của 2 từ khác nhau ('khong' vs 'bon')\n",
            "y1 = extract_mfcc(trim_endpoint(load_audio(DATASET_DIR / 'khong' / 'khong_01.wav'))[0])\n",
            "y2 = extract_mfcc(trim_endpoint(load_audio(DATASET_DIR / 'bon' / 'bon_01.wav'))[0])\n",
            "\n",
            "fig, axes = plt.subplots(2, 1, figsize=(10, 6))\n",
            "im1 = axes[0].imshow(y1.T, origin='lower', aspect='auto', cmap='magma')\n",
            "axes[0].set_title(f\"MFCC Heatmap từ 'không' (Shape: {y1.shape})\", fontweight='bold')\n",
            "fig.colorbar(im1, ax=axes[0])\n",
            "im2 = axes[1].imshow(y2.T, origin='lower', aspect='auto', cmap='magma')\n",
            "axes[1].set_title(f\"MFCC Heatmap từ 'bốn' (Shape: {y2.shape})\", fontweight='bold')\n",
            "axes[1].set_xlabel(\"Chỉ số Frame\"); fig.colorbar(im2, ax=axes[1])\n",
            "plt.tight_layout(); plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phần E: Thuật toán DTW Tự Cài Đặt (Dynamic Time Warping)\n",
            "- Khoảng cách cục bộ Euclid: $C[i, j] = \\|X[i] - Y[j]\\|_2$\n",
            "- Quy hoạch động (DP) tích lũy chi phí:\n",
            "  $$D[i, j] = C[i-1, j-1] + \\min(D[i-1, j], D[i, j-1], D[i-1, j-1])$$\n",
            "- Chuẩn hóa độ dài đường đi: $DTW_{\\text{norm}} = D[N, M] / |\\mathcal{P}|$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def dtw_custom(X, Y):\n",
            "    N, M = len(X), len(Y)\n",
            "    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]\n",
            "    C = np.sqrt(np.sum(diff ** 2, axis=-1))\n",
            "    D = np.full((N + 1, M + 1), np.inf)\n",
            "    D[0, 0] = 0.0\n",
            "    back = np.zeros((N + 1, M + 1, 2), dtype=int)\n",
            "    for i in range(1, N + 1):\n",
            "        for j in range(1, M + 1):\n",
            "            choices = [\n",
            "                (D[i - 1, j], i - 1, j),\n",
            "                (D[i, j - 1], i, j - 1),\n",
            "                (D[i - 1, j - 1], i - 1, j - 1)\n",
            "            ]\n",
            "            best, pi, pj = min(choices, key=lambda z: z[0])\n",
            "            D[i, j] = C[i - 1, j - 1] + best\n",
            "            back[i, j] = (pi, pj)\n",
            "    path = []\n",
            "    curr_i, curr_j = N, M\n",
            "    while curr_i > 0 or curr_j > 0:\n",
            "        path.append((curr_i - 1, curr_j - 1))\n",
            "        curr_i, curr_j = back[curr_i, curr_j]\n",
            "    path.reverse()\n",
            "    return D[N, M] / max(len(path), 1), path, D[1:, 1:]\n",
            "\n",
            "# Kiểm chứng cùng từ vs khác từ\n",
            "X = extract_mfcc(trim_endpoint(load_audio(DATASET_DIR / 'khong' / 'khong_01.wav'))[0])\n",
            "Y_same = extract_mfcc(trim_endpoint(load_audio(DATASET_DIR / 'khong' / 'khong_02.wav'))[0])\n",
            "Y_diff = extract_mfcc(trim_endpoint(load_audio(DATASET_DIR / 'mot' / 'mot_01.wav'))[0])\n",
            "\n",
            "cost_s, path_s, D_s = dtw_custom(X, Y_same)\n",
            "cost_d, path_d, D_d = dtw_custom(X, Y_diff)\n",
            "\n",
            "print(f\"Chi phí DTW_norm cùng từ ('khong' vs 'khong'): {cost_s:.4f}\")\n",
            "print(f\"Chi phí DTW_norm khác từ ('khong' vs 'mot'):   {cost_d:.4f} (Gấp {cost_d/cost_s:.2f} lần)\")\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "axes[0].imshow(D_s, origin='lower', cmap='viridis', aspect='auto')\n",
            "px, py = zip(*path_s); axes[0].plot(py, px, 'r-', lw=2, label='Optimal Path')\n",
            "axes[0].set_title(f\"Cùng từ: 'không' vs 'không' (Cost={cost_s:.2f})\", fontweight='bold')\n",
            "axes[0].legend()\n",
            "axes[1].imshow(D_d, origin='lower', cmap='viridis', aspect='auto')\n",
            "px_d, py_d = zip(*path_d); axes[1].plot(py_d, px_d, 'r-', lw=2, label='Optimal Path')\n",
            "axes[1].set_title(f\"Khác từ: 'không' vs 'một' (Cost={cost_d:.2f})\", fontweight='bold')\n",
            "axes[1].legend()\n",
            "plt.tight_layout(); plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phần F & G: Bộ Nhận dạng Nearest-Template & Đánh giá Toàn diện\n",
            "- Huấn luyện từ điển Template với 3 mẫu/từ.\n",
            "- Dự đoán trên 10 mẫu test độc lập.\n",
            "- Đánh giá bằng Accuracy và Confusion Matrix."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Xây dựng Template Bank\n",
            "templates = {lab: [] for lab in LABELS}\n",
            "for lab in LABELS:\n",
            "    for f in sorted((DATASET_DIR / lab).glob('*.wav'))[:3]:\n",
            "        y_t, _ = trim_endpoint(load_audio(f))\n",
            "        templates[lab].append(extract_mfcc(y_t))\n",
            "\n",
            "y_true, y_pred = [], []\n",
            "print(f\"{'Tệp test':<15} | {'Nhãn thật':<10} | {'Dự đoán':<10} | {'Top-1 Dist':<11} | {'Top-2 Dist'}\")\n",
            "print('-' * 65)\n",
            "for lab in LABELS:\n",
            "    for f in sorted((DATASET_DIR / lab).glob('*.wav'))[3:]:\n",
            "        y_t, _ = trim_endpoint(load_audio(f))\n",
            "        X_test = extract_mfcc(y_t)\n",
            "        scores = {l: min(dtw_custom(X_test, ref)[0] for ref in refs) for l, refs in templates.items()}\n",
            "        sorted_s = sorted(scores.items(), key=lambda kv: kv[1])\n",
            "        pred = sorted_s[0][0]\n",
            "        y_true.append(lab); y_pred.append(pred)\n",
            "        print(f\"{f.name:<15} | {lab:<10} | {pred:<10} | {sorted_s[0][1]:6.3f} ({pred}) | {sorted_s[1][1]:6.3f} ({sorted_s[1][0]})\")\n",
            "\n",
            "acc = accuracy_score(y_true, y_pred) * 100.0\n",
            "cm = confusion_matrix(y_true, y_pred, labels=LABELS)\n",
            "print(f\"\\nĐỘ CHÍNH XÁC ACCURACY: {acc:.1f}%\")\n",
            "\n",
            "# Vẽ Confusion Matrix\n",
            "plt.figure(figsize=(6, 5))\n",
            "plt.imshow(cm, cmap='Blues')\n",
            "plt.title(f\"Confusion Matrix - Baseline DTW\\nAccuracy = {acc:.1f}%\", fontweight='bold')\n",
            "plt.colorbar(); plt.xticks(np.arange(5), LABELS); plt.yticks(np.arange(5), LABELS)\n",
            "for i in range(5):\n",
            "    for j in range(5):\n",
            "        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')\n",
            "plt.xlabel('Nhãn Dự đoán'); plt.ylabel('Nhãn Thực tế'); plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Thí nghiệm Bắt buộc E1: Có Trim vs. Không Trim Endpoint\n",
            "Khảo sát tác động tiêu cực của khoảng lặng nền lên thuật toán căn chỉnh DTW."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "templates_notrim = {lab: [] for lab in LABELS}\n",
            "for lab in LABELS:\n",
            "    for f in sorted((DATASET_DIR / lab).glob('*.wav'))[:3]:\n",
            "        templates_notrim[lab].append(extract_mfcc(load_audio(f)))\n",
            "\n",
            "y_pred_notrim = []\n",
            "for lab in LABELS:\n",
            "    for f in sorted((DATASET_DIR / lab).glob('*.wav'))[3:]:\n",
            "        X_test = extract_mfcc(load_audio(f))\n",
            "        scores = {l: min(dtw_custom(X_test, ref)[0] for ref in refs) for l, refs in templates_notrim.items()}\n",
            "        y_pred_notrim.append(min(scores, key=scores.get))\n",
            "\n",
            "acc_notrim = accuracy_score(y_true, y_pred_notrim) * 100.0\n",
            "print(f\"Thí nghiệm E1: Accuracy khi KHÔNG trim endpoint = {acc_notrim:.1f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Thí nghiệm Bắt buộc E2: 13 MFCC chuẩn vs. 26 MFCC + Delta\n",
            "Bổ sung thông tin động học âm học (vận tốc biến thiên các formant qua thời gian)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "templates_delta = {lab: [] for lab in LABELS}\n",
            "for lab in LABELS:\n",
            "    for f in sorted((DATASET_DIR / lab).glob('*.wav'))[:3]:\n",
            "        templates_delta[lab].append(extract_mfcc(trim_endpoint(load_audio(f))[0], with_delta=True))\n",
            "\n",
            "margins = []\n",
            "for lab in LABELS:\n",
            "    for f in sorted((DATASET_DIR / lab).glob('*.wav'))[3:]:\n",
            "        X_test = extract_mfcc(trim_endpoint(load_audio(f))[0], with_delta=True)\n",
            "        scores = {l: min(dtw_custom(X_test, ref)[0] for ref in refs) for l, refs in templates_delta.items()}\n",
            "        sorted_s = sorted(scores.items(), key=lambda kv: kv[1])\n",
            "        margins.append(sorted_s[1][1] / sorted_s[0][1])\n",
            "\n",
            "print(f\"Thí nghiệm E2: Tỷ số biên độ phân tách trung bình (Top-2 / Top-1 margin): {np.mean(margins):.2f}x\")\n",
            "plt.figure(figsize=(7, 3.5))\n",
            "plt.bar(np.arange(len(margins)), margins, color='#27ae60', edgecolor='#2c3e50')\n",
            "plt.axhline(1.0, color='r', linestyle='--', label='Ngưỡng ranh giới (1.0)')\n",
            "plt.title(\"Tỷ số phân tách khoảng cách khi có Delta (MFCC + Delta)\", fontweight='bold')\n",
            "plt.xlabel(\"Mẫu test\"); plt.ylabel(\"Tỷ số\"); plt.legend(); plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Thí nghiệm Mở rộng E4: Same-Speaker vs. Cross-Speaker\n",
            "Kiểm nghiệm độ nhạy của hệ thống đối sánh mẫu DTW khi nhận dạng giọng của người nói khác (Nam -> Nữ)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if DATASET_SPK2_DIR.exists():\n",
            "    y_true_cross, y_pred_cross, cross_dists = [], [], []\n",
            "    for lab in LABELS:\n",
            "        for f in sorted((DATASET_SPK2_DIR / lab).glob('*.wav'))[3:]:\n",
            "            y_t, _ = trim_endpoint(load_audio(f))\n",
            "            X_test = extract_mfcc(y_t)\n",
            "            scores = {l: min(dtw_custom(X_test, ref)[0] for ref in refs) for l, refs in templates.items()}\n",
            "            pred = min(scores, key=scores.get)\n",
            "            y_true_cross.append(lab); y_pred_cross.append(pred)\n",
            "            cross_dists.append(scores[pred])\n",
            "    acc_cross = accuracy_score(y_true_cross, y_pred_cross) * 100.0\n",
            "    print(f\"Thí nghiệm E4: Accuracy Cross-Speaker (Template Nam -> Test Nữ) = {acc_cross:.1f}%\")\n",
            "    print(f\"Khoảng cách DTW trung bình khi khác người nói: {np.mean(cross_dists):.2f}\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path = BASE_DIR / "Lab02_Analysis.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"Đã tạo thành công Jupyter Notebook tại: {nb_path}")
