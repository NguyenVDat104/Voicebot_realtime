"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | TRƯỜNG ĐẠI HỌC THỦY LỢI
KHOA CÔNG NGHỆ THÔNG TIN - BỘ MÔN TRÍ TUỆ NHÂN TẠO

LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
Tệp thực thi toàn diện: run_lab3.py

Nội dung bao phủ trọn vẹn:
  - Phần A: Chuẩn hóa dữ liệu & Trích xuất đặc trưng 39-D (MFCC + Delta + Delta^2 + CMVN)
  - Phần B: Khởi tạo mô hình HMM Trái-sang-phải (Left-to-Right topology, Linear Temporal Segmentation)
  - Phần C: Huấn luyện Baum-Welch (EM) cho 5 mô hình từ ('mo', 'dong', 'len', 'xuong', 'dung')
  - Phần D: Nhận dạng Maximum Likelihood & Căn chỉnh trạng thái Viterbi (Trellis)
  - Phần E: Đối chuẩn thực nghiệm với DTW Baseline (Lab 2)
  - Phần F: Đọc và phân tích Pronunciation Lexicon (lexicon.txt)
  - Phần G: Xây dựng Bigram Language Model (corpus.txt) & Thử nghiệm MAP Re-ranking
  - Phần H: Bộ 6 thí nghiệm bắt buộc (E1 -> E6) & Đề xuất cơ chế Reject Out-of-Vocabulary
===============================================================================
"""

import os
import sys
import time
import math
from pathlib import Path
import joblib
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from hmmlearn.hmm import GaussianHMM

# Giảm thiểu cảnh báo MKL KMeans trên Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Cấu hình giao diện đồ thị Matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = True

# Các hằng số âm học chuẩn hóa theo Lab 3
FS = 16000                 # Tần số lấy mẫu: 16 kHz
FRAME_MS = 25              # Độ dài khung: 25 ms
HOP_MS = 10                # Bước dịch khung: 10 ms
WIN_LEN = int(FS * FRAME_MS / 1000)  # 400 mẫu
HOP_LEN = int(FS * HOP_MS / 1000)    # 160 mẫu
N_FFT = 512
N_MELS = 26
N_MFCC = 13
PRE_EMPH = 0.97
TOP_DB = 30                # Ngưỡng VAD cắt silence

LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']
VIET_MAP = {'mo': 'mở', 'dong': 'đóng', 'len': 'lên', 'xuong': 'xuống', 'dung': 'dừng'}

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
SPK2_DIR = DATASET_DIR / "test_speaker2"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LEXICON_PATH = BASE_DIR / "lexicon.txt"
CORPUS_PATH = BASE_DIR / "corpus.txt"
RESULTS_CSV = BASE_DIR / "results.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PHẦN A: TRÍCH XUẤT ĐẶC TRƯNG 39 CHIỀU & CHUẨN HÓA CMVN
# =============================================================================

def extract_features(path_or_audio, sr=FS, use_dynamic=True, apply_cmvn=True):
    """
    Trích xuất đặc trưng MFCC 39 chiều (13 MFCC + 13 Delta + 13 Delta-Delta)
    kèm tiền xử lý VAD trim và Utterance-level CMVN.
    """
    if isinstance(path_or_audio, (str, Path)):
        y, orig_sr = librosa.load(str(path_or_audio), sr=sr, mono=True)
    else:
        y = path_or_audio.astype(np.float32)

    # 1. VAD cắt khoảng lặng
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y_trimmed) < WIN_LEN:
        y_trimmed = y

    # 2. Tiền nhấn (Pre-emphasis)
    y_pre = librosa.effects.preemphasis(y_trimmed, coef=PRE_EMPH)

    # 3. Trích 13 hệ số MFCC tĩnh
    mfcc = librosa.feature.mfcc(
        y=y_pre, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, win_length=WIN_LEN,
        hop_length=HOP_LEN, window='hamming',
        n_mels=N_MELS, fmin=20, fmax=sr/2
    )

    if not use_dynamic:
        F = mfcc.T  # Ma trận T x 13
    else:
        # 4. Tính toán đạo hàm động bậc 1 (Delta) và bậc 2 (Delta-Delta)
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        F = np.vstack([mfcc, d1, d2]).T  # Ma trận T x 39

    # 5. Utterance-level CMVN
    if apply_cmvn:
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True) + 1e-8
        F = (F - mu) / sd

    return F


def plot_feature_heatmaps():
    """Vẽ và so sánh Heatmap đặc trưng 39 chiều của 2 từ: 'mo' và 'dong'."""
    file_mo = sorted(TRAIN_DIR.glob('mo/*.wav'))[0]
    file_dong = sorted(TRAIN_DIR.glob('dong/*.wav'))[0]

    feat_mo = extract_features(file_mo)
    feat_dong = extract_features(file_dong)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    im0 = axes[0].imshow(feat_mo.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"Đặc trưng 39-D MFCC + Delta + Delta^2 (CMVN): Lệnh 'MỞ' ({file_mo.name})", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Chỉ số Chiều\n(0-12: MFCC, 13-25: Delta, 26-38: Delta^2)")
    axes[0].axhline(12.5, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].axhline(25.5, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    fig.colorbar(im0, ax=axes[0], pad=0.01, label="Normalized Value")

    im1 = axes[1].imshow(feat_dong.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f"Đặc trưng 39-D MFCC + Delta + Delta^2 (CMVN): Lệnh 'ĐÓNG' ({file_dong.name})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Chỉ số Khung thời gian (Frame Index: 10 ms/frame)")
    axes[1].set_ylabel("Chỉ số Chiều (0-38)")
    axes[1].axhline(12.5, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[1].axhline(25.5, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    fig.colorbar(im1, ax=axes[1], pad=0.01, label="Normalized Value")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_mfcc39_heatmaps.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Phần A] Đã xuất Heatmap 39-D đặc trưng: {out_path.name}")


# =============================================================================
# PHẦN B & C: KHỞI TẠO & HUẤN LUYỆN HMM TRÁI-SANG-PHẢI (BAUM-WELCH)
# =============================================================================

def train_single_hmm(label, train_files, n_states=5, covariance_type='diag', n_iter=80, use_dynamic=True, random_state=42):
    """
    Huấn luyện mô hình Left-to-Right Gaussian HMM sử dụng Linear Temporal Segmentation.
    Phân chia đều chuỗi thời gian của các utterance huấn luyện thành n_states phân đoạn,
    ước lượng means và covars ban đầu dọc theo tiến trình phát âm, sau đó áp dụng EM.
    """
    feats = [extract_features(w, use_dynamic=use_dynamic) for w in train_files]
    lengths = [len(f) for f in feats]
    X = np.vstack(feats)
    D = X.shape[1]

    # 1. Phân đoạn thời gian đều (Linear Temporal Segmentation)
    state_slices = [[] for _ in range(n_states)]
    offset = 0
    for L in lengths:
        seg_len = L / n_states
        for s in range(n_states):
            st = offset + int(round(s * seg_len))
            en = offset + int(round((s + 1) * seg_len))
            if en > st:
                state_slices[s].append(X[st:en])
        offset += L

    # Ước lượng means & covars ban đầu
    means = np.array([np.vstack(state_slices[s]).mean(axis=0) for s in range(n_states)])
    if covariance_type == 'diag':
        covars = np.array([np.vstack(state_slices[s]).var(axis=0) + 1e-2 for s in range(n_states)])
    else:
        covars = np.array([np.cov(np.vstack(state_slices[s]).T) + np.eye(D) * 1e-2 for s in range(n_states)])

    # 2. Khởi tạo Left-to-Right Topology
    start = np.zeros(n_states)
    start[0] = 1.0

    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3
    A[-1, -1] = 1.0

    # 3. Khởi tạo GaussianHMM và cập nhật qua Baum-Welch (EM)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=1e-3,
        min_covar=1e-3,
        init_params='',  # Dùng giá trị khởi tạo phân đoạn của ta
        params='tmc',    # Cập nhật transition, means, covars
        random_state=random_state
    )
    model.startprob_ = start
    model.transmat_ = A
    model.means_ = means
    model.covars_ = covars

    model.fit(X, lengths)

    # Đảm bảo ràng buộc Left-to-Right không bị rò rỉ
    for i in range(n_states):
        for j in range(n_states):
            if j < i or j > i + 1:
                model.transmat_[i, j] = 0.0
        row_sum = np.sum(model.transmat_[i])
        if row_sum > 0:
            model.transmat_[i] /= row_sum
        else:
            model.transmat_[i, i] = 1.0

    return model


def train_word_hmms(n_states=5, covariance_type='diag', use_dynamic=True, num_samples_per_class=6):
    """
    Huấn luyện 1 mô hình Gaussian HMM cho từng từ/lệnh trong tập TRAIN.
    """
    models = {}
    print(f"\n[Phần C] Huấn luyện 5 HMM ({n_states} states, cov='{covariance_type}', dynamic={use_dynamic}, K={num_samples_per_class})...")

    for label in LABELS:
        train_files = sorted(TRAIN_DIR.glob(f"{label}/*.wav"))[:num_samples_per_class]
        model = train_single_hmm(label, train_files, n_states=n_states,
                                 covariance_type=covariance_type, use_dynamic=use_dynamic)
        models[label] = model

        # Lưu trữ mô hình baseline
        if n_states == 5 and covariance_type == 'diag' and use_dynamic and num_samples_per_class == 6:
            joblib.dump(model, MODELS_DIR / f"hmm_{label}.pkl")

        is_conv = model.monitor_.converged
        n_iters = len(model.monitor_.history)
        print(f"  -> Model '{label:6s}': Hội tụ={is_conv} ({n_iters:02d} vòng lặp EM) | Transmat shape={model.transmat_.shape}")

    return models


# =============================================================================
# PHẦN D: NHẬN DẠNG MAXIMUM LIKELIHOOD & CĂN CHỈNH VITERBI
# =============================================================================

def recognize_utterance(feat, models):
    """
    Tính log-likelihood cho chuỗi đặc trưng dưới tất cả các mô hình và chọn argmax.
    """
    scores = {}
    for label, model in models.items():
        try:
            scores[label] = model.score(feat)
        except Exception:
            scores[label] = -1e9
    pred = max(scores, key=scores.get)
    return pred, scores


def evaluate_models(models, test_dir=TEST_DIR, use_dynamic=True):
    """Đánh giá toàn diện trên tập test, trả về accuracy, true_y, pred_y."""
    true_y, pred_y = [], []
    all_scores = []

    for label in LABELS:
        wav_files = sorted((test_dir / label).glob("*.wav"))
        for wav_path in wav_files:
            F = extract_features(wav_path, use_dynamic=use_dynamic)
            pred, scores = recognize_utterance(F, models)
            true_y.append(label)
            pred_y.append(pred)
            all_scores.append({
                'file': wav_path.name,
                'true': label,
                'pred': pred,
                'scores': scores,
                'T': len(F)
            })

    acc = accuracy_score(true_y, pred_y)
    return acc, true_y, pred_y, all_scores


def plot_viterbi_alignments(models):
    """Vẽ và phân tích chuỗi trạng thái Viterbi cho 2 utterance khác nhau."""
    sample_files = [
        ('mo', sorted(TEST_DIR.glob('mo/*.wav'))[0]),
        ('dong', sorted(TEST_DIR.glob('dong/*.wav'))[0])
    ]

    for label, wav_path in sample_files:
        F = extract_features(wav_path)
        logp, states = models[label].decode(F, algorithm='viterbi')

        plt.figure(figsize=(9, 4.5))
        plt.step(range(len(states)), states, where='post', color='#1f77b4', linewidth=2.5, label='Viterbi state path')
        plt.scatter(range(len(states)), states, color='#ff7f0e', s=20, zorder=3)
        plt.title(f"Viterbi State Alignment: {wav_path.name} đối với HMM '{label.upper()}' (Log-prob: {logp:.1f})", fontsize=11, fontweight='bold')
        plt.xlabel("Khung thời gian (Frame Index: 10 ms/frame)")
        plt.ylabel("Chỉ số Trạng thái (0 -> 4)")
        plt.yticks(range(5), [f"State q{i+1}" for i in range(5)])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()

        out_fig = FIGURES_DIR / f"fig{2 if label=='mo' else 3}_viterbi_alignment_{label}.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"[Phần D] Đã xuất đồ thị Viterbi Trellis alignment: {out_fig.name}")


# =============================================================================
# PHẦN E: ĐỐI CHUẨN THỰC NGHIỆM VỚI DTW BASELINE (LAB 2)
# =============================================================================

def compute_dtw_distance(feat1, feat2):
    """
    Tự cài đặt thuật toán Dynamic Time Warping (DTW) chuẩn xác từ Lab 2.
    Khoảng cách cục bộ Euclid, quy tắc bước 3 hướng, chuẩn hóa theo chiều dài đường dẫn.
    """
    N, M = len(feat1), len(feat2)
    diff = feat1[:, np.newaxis, :] - feat2[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    cost = np.full((N, M), np.inf)
    cost[0, 0] = dist[0, 0]

    for i in range(1, N):
        cost[i, 0] = cost[i - 1, 0] + dist[i, 0]
    for j in range(1, M):
        cost[0, j] = cost[0, j - 1] + dist[0, j]

    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = dist[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    i, j = N - 1, M - 1
    path_len = 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            if min_val == cost[i - 1, j - 1]:
                i, j = i - 1, j - 1
            elif min_val == cost[i - 1, j]:
                i -= 1
            else:
                j -= 1
        path_len += 1

    return cost[N - 1, M - 1] / path_len


class DTWClassifier:
    """Bộ nhận dạng dựa trên mẫu DTW (Nearest-Template) kế thừa từ Lab 2."""
    def __init__(self):
        self.templates = {}

    def fit(self, train_dir, num_samples_per_class=6, use_dynamic=True):
        self.templates = {label: [] for label in LABELS}
        t0 = time.time()
        for label in LABELS:
            files = sorted((train_dir / label).glob("*.wav"))[:num_samples_per_class]
            for wav_path in files:
                feat = extract_features(wav_path, use_dynamic=use_dynamic)
                self.templates[label].append(feat)
        self.train_time = time.time() - t0
        return self

    def predict_one(self, feat):
        best_dist = float('inf')
        best_label = None
        for label, tmpls in self.templates.items():
            for tmpl in tmpls:
                d = compute_dtw_distance(feat, tmpl)
                if d < best_dist:
                    best_dist = d
                    best_label = label
        return best_label, best_dist

    def evaluate(self, test_dir=TEST_DIR, use_dynamic=True):
        true_y, pred_y = [], []
        latencies = []
        for label in LABELS:
            for wav_path in sorted((test_dir / label).glob("*.wav")):
                feat = extract_features(wav_path, use_dynamic=use_dynamic)
                t0 = time.time()
                pred, _ = self.predict_one(feat)
                latencies.append((time.time() - t0) * 1000)
                true_y.append(label)
                pred_y.append(pred)
        acc = accuracy_score(true_y, pred_y)
        avg_lat = np.mean(latencies)
        return acc, true_y, pred_y, avg_lat


# =============================================================================
# PHẦN F & G: LEXICON & BIGRAM LANGUAGE MODEL RE-RANKING
# =============================================================================

def parse_lexicon(lexicon_file=LEXICON_PATH):
    """Đọc và giải tích tệp lexicon.txt thành cấu trúc từ điển Python."""
    lex_dict = {}
    with open(lexicon_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            word = parts[0]
            phones = parts[1:]
            lex_dict[word] = phones
    return lex_dict


def train_bigram_lm(corpus_file=CORPUS_PATH):
    """
    Xây dựng mô hình Unigram và Bigram từ tập ngữ liệu corpus.txt
    có hỗ trợ Laplace Add-k smoothing.
    """
    unigrams = {}
    bigrams = {}

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            for i, w in enumerate(tokens):
                unigrams[w] = unigrams.get(w, 0) + 1
                if i > 0:
                    pair = (tokens[i - 1], w)
                    bigrams[pair] = bigrams.get(pair, 0) + 1

    return unigrams, bigrams


def bigram_probability(prev_word, cur_word, unigrams, bigrams, vocab_size, k=0.5):
    """Tính xác suất chuyển tiếp Bigram P(cur | prev) có Add-k smoothing."""
    c_prev_cur = bigrams.get((prev_word, cur_word), 0)
    c_prev = unigrams.get(prev_word, 0)
    return (c_prev_cur + k) / (c_prev + k * vocab_size)


def map_rerank(feat, prev_word, models, unigrams, bigrams, lm_weight=0.3, k=0.5):
    """
    Kết hợp điểm số âm học chuẩn hóa và điểm số ngôn ngữ:
    Score(W) = log P_A(X|W)/T + lambda_LM * log P_L(W | W_prev)
    """
    T = len(feat)
    vocab_size = len(models)
    total_scores = {}
    acoustic_scores = {}
    lm_scores = {}

    for w, model in models.items():
        try:
            a_score = model.score(feat) / T
        except Exception:
            a_score = -1e4
        p_lm = bigram_probability(prev_word, w, unigrams, bigrams, vocab_size, k=k)
        lm_score = math.log(max(p_lm, 1e-9))

        acoustic_scores[w] = a_score
        lm_scores[w] = lm_score
        total_scores[w] = a_score + lm_weight * lm_score

    best_w = max(total_scores, key=total_scores.get)
    return best_w, total_scores, acoustic_scores, lm_scores


# =============================================================================
# PHẦN H: THÍ NGHIỆM BẮT BUỘC (E1 -> E6) & ĐÁNH GIÁ ĐỘ BỀN VỮNG (ROBUSTNESS)
# =============================================================================

def run_all_experiments(trained_models):
    """
    Thực thi trọn vẹn bộ thí nghiệm E1 -> E6 theo yêu cầu chuẩn của Lab 3.
    """
    results_records = []
    print("\n" + "=" * 65)
    print("BẮT ĐẦU CHUỖI 6 THÍ NGHIỆM BẮT BUỘC (E1 -> E6)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Thí nghiệm E1: Khảo sát số trạng thái n_states in {3, 5, 7}
    # -------------------------------------------------------------------------
    print("\n[Thí nghiệm E1] Khảo sát số lượng trạng thái HMM: n_states in {3, 5, 7}...")
    for ns in [3, 5, 7]:
        t0 = time.time()
        m_e1 = train_word_hmms(n_states=ns, covariance_type='diag', use_dynamic=True)
        acc_e1, _, _, _ = evaluate_models(m_e1, TEST_DIR, use_dynamic=True)
        train_time = time.time() - t0
        print(f"  -> n_states={ns}: Accuracy = {acc_e1 * 100:.1f}% | Train Time = {train_time:.2f}s")
        results_records.append({
            'Experiment': 'E1_Number_of_States',
            'Variable': f'n_states={ns}',
            'Accuracy': acc_e1,
            'Detail': f'Train_Time={train_time:.2f}s'
        })

    # -------------------------------------------------------------------------
    # Thí nghiệm E2: 13 MFCC tĩnh vs 39 MFCC động
    # -------------------------------------------------------------------------
    print("\n[Thí nghiệm E2] Khảo sát ảnh hưởng của đặc trưng động: 13 MFCC vs 39-D...")
    m_13 = train_word_hmms(n_states=5, covariance_type='diag', use_dynamic=False)
    acc_13, _, _, _ = evaluate_models(m_13, TEST_DIR, use_dynamic=False)
    acc_39, _, _, _ = evaluate_models(trained_models, TEST_DIR, use_dynamic=True)
    print(f"  -> 13 MFCC tĩnh: Accuracy = {acc_13 * 100:.1f}%")
    print(f"  -> 39-D MFCC động: Accuracy = {acc_39 * 100:.1f}%")
    results_records.append({'Experiment': 'E2_Dynamic_Features', 'Variable': '13_Static_MFCC', 'Accuracy': acc_13, 'Detail': 'Dim=13'})
    results_records.append({'Experiment': 'E2_Dynamic_Features', 'Variable': '39_Dynamic_MFCC', 'Accuracy': acc_39, 'Detail': 'Dim=39'})

    # -------------------------------------------------------------------------
    # Thí nghiệm E3: Ma trận hiệp phương sai Diagonal vs Full
    # -------------------------------------------------------------------------
    print("\n[Thí nghiệm E3] Khảo sát cấu trúc hiệp phương sai: Diag vs Full...")
    try:
        m_full = train_word_hmms(n_states=5, covariance_type='full', use_dynamic=True)
        acc_full, _, _, _ = evaluate_models(m_full, TEST_DIR, use_dynamic=True)
        print(f"  -> Full Covariance: Accuracy = {acc_full * 100:.1f}%")
    except Exception as e:
        print(f"  -> Full Covariance lỗi hội tụ/singularity: {e}")
        acc_full = 0.0

    print(f"  -> Diag Covariance: Accuracy = {acc_39 * 100:.1f}%")
    results_records.append({'Experiment': 'E3_Covariance_Type', 'Variable': 'Diagonal', 'Accuracy': acc_39, 'Detail': 'Params=39/state'})
    results_records.append({'Experiment': 'E3_Covariance_Type', 'Variable': 'Full', 'Accuracy': acc_full, 'Detail': 'Params=780/state'})

    # -------------------------------------------------------------------------
    # Thí nghiệm E4: Đối chuẩn chi tiết DTW vs HMM qua kích thước tập huấn luyện
    # -------------------------------------------------------------------------
    print("\n[Thí nghiệm E4] Đối chuẩn DTW vs HMM với số lượng mẫu huấn luyện K in {1, 2, 3, 6}...")
    dtw_accs, hmm_accs = [], []
    k_samples = [1, 2, 3, 6]

    for k in k_samples:
        dtw_clf = DTWClassifier()
        dtw_clf.fit(TRAIN_DIR, num_samples_per_class=k, use_dynamic=True)
        acc_dtw_k, _, _, _ = dtw_clf.evaluate(TEST_DIR, use_dynamic=True)
        dtw_accs.append(acc_dtw_k)

        m_k = train_word_hmms(n_states=5, covariance_type='diag', use_dynamic=True, num_samples_per_class=k)
        acc_hmm_k, _, _, _ = evaluate_models(m_k, TEST_DIR, use_dynamic=True)
        hmm_accs.append(acc_hmm_k)

        print(f"  -> K={k} mẫu/lớp: DTW Accuracy = {acc_dtw_k * 100:.1f}% | HMM Accuracy = {acc_hmm_k * 100:.1f}%")
        results_records.append({'Experiment': 'E4_Sample_Size', 'Variable': f'K={k}_DTW', 'Accuracy': acc_dtw_k, 'Detail': f'K={k}'})
        results_records.append({'Experiment': 'E4_Sample_Size', 'Variable': f'K={k}_HMM', 'Accuracy': acc_hmm_k, 'Detail': f'K={k}'})

    # Vẽ biểu đồ so sánh DTW vs HMM
    plt.figure(figsize=(8, 4.5))
    plt.plot(k_samples, [a * 100 for a in dtw_accs], marker='o', linewidth=2.2, label='DTW Baseline (Lab 2)', color='#d62728')
    plt.plot(k_samples, [a * 100 for a in hmm_accs], marker='s', linewidth=2.2, label='Gaussian HMM (Lab 3)', color='#1f77b4')
    plt.title("So sánh Hiệu năng Nhận dạng: DTW vs Gaussian HMM theo Số mẫu Train", fontsize=11, fontweight='bold')
    plt.xlabel("Số mẫu Huấn luyện trên mỗi Lớp (K)")
    plt.ylabel("Độ chính xác (Accuracy %)")
    plt.xticks(k_samples)
    plt.ylim(0, 105)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_dtw_vs_hmm_comparison.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Thí nghiệm E5: Kiểm thử Mismatch (Speaker 2 & Additive Noise)
    # -------------------------------------------------------------------------
    print("\n[Thí nghiệm E5] Đánh giá độ bền vững trước điều kiện Mismatch...")
    acc_clean, _, _, _ = evaluate_models(trained_models, TEST_DIR, use_dynamic=True)
    acc_spk2, _, _, _ = evaluate_models(trained_models, SPK2_DIR, use_dynamic=True)

    rng = np.random.default_rng(42)
    true_noisy, pred_noisy = [], []
    for label in LABELS:
        for wav_path in sorted((TEST_DIR / label).glob("*.wav")):
            y, sr = librosa.load(str(wav_path), sr=FS, mono=True)
            sig_power = np.mean(y ** 2)
            noise_power = sig_power / (10 ** (15 / 10))
            noise = rng.normal(0, np.sqrt(noise_power), len(y))
            y_noisy = y + noise.astype(np.float32)

            F_noisy = extract_features(y_noisy, use_dynamic=True)
            pred_n, _ = recognize_utterance(F_noisy, trained_models)
            true_noisy.append(label)
            pred_noisy.append(pred_n)
    acc_noisy = accuracy_score(true_noisy, pred_noisy)

    print(f"  -> Sạch (Clean Speaker 1): Accuracy = {acc_clean * 100:.1f}%")
    print(f"  -> Cross-Speaker (Speaker 2): Accuracy = {acc_spk2 * 100:.1f}% (Δ = {(acc_clean - acc_spk2)*100:+.1f}%)")
    print(f"  -> Nhiễu môi trường (SNR 15dB): Accuracy = {acc_noisy * 100:.1f}% (Δ = {(acc_clean - acc_noisy)*100:+.1f}%)")
    results_records.append({'Experiment': 'E5_Robustness', 'Variable': 'Clean_Test', 'Accuracy': acc_clean, 'Detail': 'Matched'})
    results_records.append({'Experiment': 'E5_Robustness', 'Variable': 'Cross_Speaker2', 'Accuracy': acc_spk2, 'Detail': 'Female Voice'})
    results_records.append({'Experiment': 'E5_Robustness', 'Variable': 'Noise_SNR15dB', 'Accuracy': acc_noisy, 'Detail': 'White Noise'})

    # -------------------------------------------------------------------------
    # Thí nghiệm E6: Khảo sát trọng số Language Model lambda_LM
    # -------------------------------------------------------------------------
    print("\n[Thí nghiệm E6] Khảo sát trọng số Mô hình ngôn ngữ lambda_LM in {0.0, 0.1, 0.3, 1.0}...")
    unigrams, bigrams = train_bigram_lm(CORPUS_PATH)
    prev_context = "yeu cau"
    target_wav = sorted((TEST_DIR / "mo").glob("*.wav"))[0]
    F_test = extract_features(target_wav, use_dynamic=True)

    lm_weights = [0.0, 0.1, 0.3, 1.0]
    rerank_results = {}
    for lw in lm_weights:
        best_w, total_scores, ac_scores, lm_scores = map_rerank(
            F_test, prev_context, trained_models, unigrams, bigrams, lm_weight=lw
        )
        rerank_results[lw] = (best_w, total_scores)
        print(f"  -> lambda_LM = {lw:3.1f} | Top-1 Quyết định: '{best_w.upper()}' | Điểm số: {total_scores[best_w]:.2f}")
        results_records.append({
            'Experiment': 'E6_LM_Weight',
            'Variable': f'lambda={lw}',
            'Accuracy': 1.0 if best_w == 'mo' else 0.0,
            'Detail': f'Winner={best_w}'
        })

    # Vẽ biểu đồ phân tích E6
    plt.figure(figsize=(9, 4.5))
    x_indices = np.arange(len(LABELS))
    bar_width = 0.2
    for idx, lw in enumerate(lm_weights):
        _, total_s = rerank_results[lw]
        scores_list = [total_s[lbl] for lbl in LABELS]
        plt.bar(x_indices + idx * bar_width, scores_list, width=bar_width, label=f'λ_LM = {lw}')
    plt.title(f"MAP Re-ranking Điểm số Tổng hợp theo Trọng số λ_LM (Ngữ cảnh: '{prev_context}')", fontsize=11, fontweight='bold')
    plt.xlabel("Từ vựng Ứng viên (Candidate Words)")
    plt.ylabel("Normalized Combined Score")
    plt.xticks(x_indices + 1.5 * bar_width, [lbl.upper() for lbl in LABELS])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_lm_reranking_analysis.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Cơ chế Reject Out-of-Vocabulary / Unknown
    # -------------------------------------------------------------------------
    print("\n[Mục H] Phân tích phân bố điểm số và thiết lập ngưỡng Từ chối (Reject Threshold)...")
    correct_scores = []
    incorrect_scores = []
    margins = []

    for label in LABELS:
        for wav_path in sorted((TEST_DIR / label).glob("*.wav")):
            feat = extract_features(wav_path, use_dynamic=True)
            T = len(feat)
            scores = {lbl: trained_models[lbl].score(feat) / T for lbl in LABELS}
            sorted_s = sorted(scores.values(), reverse=True)
            top1_lbl = max(scores, key=scores.get)

            if top1_lbl == label:
                correct_scores.append(sorted_s[0])
            else:
                incorrect_scores.append(sorted_s[0])
            margins.append(sorted_s[0] - sorted_s[1])

    suggested_threshold = np.mean(correct_scores) - 2.0 * np.std(correct_scores)
    suggested_margin = 0.15
    print(f"  -> Normalized Score Đúng lớp: Mean = {np.mean(correct_scores):.2f}, Std = {np.std(correct_scores):.2f}")
    print(f"  -> Đề xuất Ngưỡng Điểm / Frame (Score Threshold): {suggested_threshold:.2f}")
    print(f"  -> Đề xuất Ngưỡng Khoảng cách Top-1 & Top-2 (Margin Threshold): {suggested_margin:.2f}")

    plt.figure(figsize=(8, 4.5))
    plt.hist(correct_scores, bins=8, alpha=0.7, color='#2ca02c', label='Đúng lớp (In-vocabulary)')
    if incorrect_scores:
        plt.hist(incorrect_scores, bins=5, alpha=0.7, color='#d62728', label='Sai lớp')
    plt.axvline(suggested_threshold, color='black', linestyle='--', linewidth=1.8, label=f'Reject Threshold ({suggested_threshold:.2f})')
    plt.title("Phân bố Điểm số Chuẩn hóa theo Frame (Normalized Log-likelihood/T)", fontsize=11, fontweight='bold')
    plt.xlabel("Score / Frame (log P(X|λ) / T)")
    plt.ylabel("Số lượng mẫu")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_score_distributions_reject.png", dpi=300)
    plt.close()

    # Ghi toàn bộ kết quả thí nghiệm ra results.csv
    with open(RESULTS_CSV, 'w', encoding='utf-8') as f:
        f.write("Experiment,Variable,Accuracy,Detail\n")
        for rec in results_records:
            f.write(f"{rec['Experiment']},{rec['Variable']},{rec['Accuracy']:.4f},{rec['Detail']}\n")
    print(f"[Hoàn tất] Toàn bộ kết quả đã được lưu tại: {RESULTS_CSV.name}")


# =============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# =============================================================================

def main():
    print("=" * 70)
    print("KHỞI CHẠY PIPELINE LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM")
    print("Khoa CNTT - Bộ môn Trí tuệ Nhân tạo - Trường Đại học Thủy lợi")
    print("=" * 70)

    # 1. Vẽ Heatmap đặc trưng 39-D
    plot_feature_heatmaps()

    # 2. Huấn luyện 5 mô hình HMM baseline
    trained_models = train_word_hmms(n_states=5, covariance_type='diag', use_dynamic=True)

    # 3. Đánh giá trên tập test sạch
    acc_clean, true_y, pred_y, all_scores = evaluate_models(trained_models, TEST_DIR, use_dynamic=True)
    print(f"\n[Phần D] Độ chính xác Baseline HMM (Tập Test Sạch): {acc_clean * 100:.2f}%")

    # 4. Xuất ma trận nhầm lẫn Confusion Matrix
    cm = confusion_matrix(true_y, pred_y, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[lbl.upper() for lbl in LABELS])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f"Confusion Matrix: HMM Baseline (Acc = {acc_clean * 100:.1f}%)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = FIGURES_DIR / "fig4_confusion_matrix_hmm.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Phần D] Đã xuất Confusion Matrix: {cm_path.name}")

    # 5. Vẽ Viterbi State Alignment Trellis
    plot_viterbi_alignments(trained_models)

    # 6. Bảng điểm âm học chi tiết của 3 utterance kiểm thử
    print("\n" + "-" * 70)
    print("BẢNG ĐIỂM ACOUSTIC LOG-LIKELIHOOD CỦA 3 TEST UTTERANCES VỚI 5 MÔ HÌNH:")
    print("-" * 70)
    header = f"{'File':<16} | {'True':<6} | {'Pred':<6} | " + " | ".join([f"{lbl.upper():<8}" for lbl in LABELS])
    print(header)
    print("-" * len(header))
    for sample in all_scores[:3]:
        scores_str = " | ".join([f"{sample['scores'][lbl]:8.1f}" for lbl in LABELS])
        print(f"{sample['file']:<16} | {sample['true']:<6} | {sample['pred']:<6} | {scores_str}")
    print("-" * 70)

    # 7. Đọc và xác thực Pronunciation Lexicon
    print("\n[Phần F] Đọc và giải thích Lexicon (lexicon.txt):")
    lexicon = parse_lexicon(LEXICON_PATH)
    for w, phones in lexicon.items():
        print(f"  Word: {w:6s} -> Phones: {' - '.join(phones)}")

    # 8. Chạy toàn bộ 6 thí nghiệm bắt buộc E1 -> E6
    run_all_experiments(trained_models)

    print("\n" + "=" * 70)
    print("HOÀN TẤT THỰC THI TOÀN BỘ PIPELINE LAB 3 THÀNH CÔNG RỰC RỠ!")
    print(f"  - Các mô hình đã lưu tại: {MODELS_DIR}")
    print(f"  - Các đồ thị đã xuất tại: {FIGURES_DIR}")
    print(f"  - Bảng số liệu định lượng: {RESULTS_CSV}")
    print("=" * 70)


if __name__ == '__main__':
    main()
