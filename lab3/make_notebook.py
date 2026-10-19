"""
Script tự động sinh Jupyter Notebook Lab03.ipynb chuẩn định dạng v4
Phục vụ bài thực hành CSE457 - Lab 3: Nhận dạng tiếng nói nâng cao với HMM
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 3\n",
            "## NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM (HIDDEN MARKOV MODEL)\n",
            "**Trường Đại học Thủy Lợi - Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo**\n",
            "\n",
            "---\n",
            "### Thông tin sinh viên\n",
            "- **Họ và tên:** Nguyễn Văn Đạt\n",
            "- **Mã sinh viên:** 215106xxxx\n",
            "- **Lớp:** CSE457\n",
            "- **Học phần:** Xử lý âm thanh và tiếng nói\n",
            "- **Nhiệm vụ:** Nhận dạng 5 lệnh tiếng Việt (`mở`, `đóng`, `lên`, `xuống`, `dừng`) bằng Gaussian HMM\n",
            "\n",
            "---\n",
            "### Mục tiêu cốt lõi của bài Lab 3\n",
            "1. **Chuyển dịch mô hình:** Từ so khớp mẫu DTW ở Lab 2 sang mô hình xác suất theo thời gian Gaussian HMM.\n",
            "2. **Đặc trưng 39 chiều:** Trích xuất 13 MFCC tĩnh kết hợp $\\Delta$ (vận tốc) và $\\Delta^2$ (gia tốc) cùng chuẩn hóa Utterance-level CMVN.\n",
            "3. **Kiến trúc HMM Trái-sang-phải (Left-to-Right):** Xây dựng topology Bakis 5 trạng thái, phân bố Gaussian đường chéo (`diag`).\n",
            "4. **Ba bài toán cổ điển HMM:**\n",
            "   - **Evaluation / Likelihood:** Tính $P(X|\\lambda)$ qua thuật toán Forward.\n",
            "   - **Decoding / Alignment:** Tìm đường đi trạng thái tối ưu qua thuật toán Viterbi.\n",
            "   - **Training / Learning:** Ước lượng tham số qua giải thuật Baum-Welch (EM).\n",
            "5. **Đối chuẩn thực nghiệm với DTW Baseline:** So sánh độ chính xác, tốc độ, tài nguyên và bản chất căn chỉnh thời gian.\n",
            "6. **Mở rộng Lexicon & Bigram LM:** Tích hợp Pronunciation Lexicon và Mô hình ngôn ngữ Bigram làm mịn Add-k theo nguyên lý Bayes (MAP Re-ranking).\n",
            "7. **Bộ 6 thí nghiệm bắt buộc (E1 $\\rightarrow$ E6):** Khảo sát số trạng thái, đặc trưng động, cấu trúc hiệp phương sai, số mẫu huấn luyện, mismatch môi trường/người nói và trọng số $\\lambda_{LM}$."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. KHỞI TẠO MÔI TRƯỜNG VÀ CẤU HÌNH THAM SỐ ÂM HỌC"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import time\n",
            "import math\n",
            "from pathlib import Path\n",
            "import joblib\n",
            "import numpy as np\n",
            "import soundfile as sf\n",
            "import librosa\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay\n",
            "from hmmlearn.hmm import GaussianHMM\n",
            "\n",
            "# Thiết lập môi trường và cấu hình đồ thị\n",
            "os.environ[\"OMP_NUM_THREADS\"] = \"1\"\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
            "plt.rcParams['figure.autolayout'] = True\n",
            "\n",
            "# Các hằng số âm học chuẩn hóa theo đề bài Lab 3\n",
            "FS = 16000                 # Tần số lấy mẫu: 16 kHz\n",
            "FRAME_MS = 25              # Độ dài khung: 25 ms (400 mẫu)\n",
            "HOP_MS = 10                # Bước dịch khung: 10 ms (160 mẫu)\n",
            "WIN_LEN = int(FS * FRAME_MS / 1000)\n",
            "HOP_LEN = int(FS * HOP_MS / 1000)\n",
            "N_FFT = 512\n",
            "N_MELS = 26\n",
            "N_MFCC = 13\n",
            "PRE_EMPH = 0.97\n",
            "TOP_DB = 30                # Ngưỡng VAD cắt khoảng lặng\n",
            "\n",
            "LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']\n",
            "VIET_MAP = {'mo': 'mở', 'dong': 'đóng', 'len': 'lên', 'xuong': 'xuống', 'dung': 'dừng'}\n",
            "\n",
            "# Đường dẫn thư mục\n",
            "BASE_DIR = Path(\".\").resolve()\n",
            "TRAIN_DIR = BASE_DIR / \"dataset\" / \"train\"\n",
            "TEST_DIR = BASE_DIR / \"dataset\" / \"test\"\n",
            "SPK2_DIR = BASE_DIR / \"dataset\" / \"test_speaker2\"\n",
            "FIGURES_DIR = BASE_DIR / \"figures\"\n",
            "MODELS_DIR = BASE_DIR / \"models\"\n",
            "\n",
            "print(f\"Thư mục làm việc: {BASE_DIR}\")\n",
            "print(f\"Tập nhãn từ vựng (5 lệnh): {LABELS}\")\n",
            "print(f\"Số file TRAIN: {len(list(TRAIN_DIR.glob('*/*.wav')))} | TEST: {len(list(TEST_DIR.glob('*/*.wav')))}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. PHẦN A: TRÍCH XUẤT ĐẶC TRƯNG 39 CHIỀU VÀ CHUẨN HÓA CMVN\n",
            "Vector đặc trưng tại mỗi khung thời gian $t$ là sự kết hợp của:\n",
            "$$x_t = [c_t^T, \\Delta c_t^T, \\Delta^2 c_t^T]^T \\in \\mathbb{R}^{39}$$\n",
            "- $c_t \\in \\mathbb{R}^{13}$: 13 hệ số MFCC mô tả hình dạng bao phổ ngắn hạn.\n",
            "- $\\Delta c_t \\in \\mathbb{R}^{13}$: Đạo hàm bậc 1 mô tả xu hướng vận tốc thay đổi phổ.\n",
            "- $\\Delta^2 c_t \\in \\mathbb{R}^{13}$: Đạo hàm bậc 2 mô tả gia tốc biến thiên phổ.\n",
            "- Chuẩn hóa **Utterance-level CMVN**: $\\hat{x}_t = \\frac{x_t - \\mu_X}{\\sigma_X + 10^{-8}}$ nhằm triệt tiêu méo dạng kênh truyền."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def extract_features(path_or_audio, sr=FS, use_dynamic=True, apply_cmvn=True):\n",
            "    if isinstance(path_or_audio, (str, Path)):\n",
            "        y, orig_sr = librosa.load(str(path_or_audio), sr=sr, mono=True)\n",
            "    else:\n",
            "        y = path_or_audio.astype(np.float32)\n",
            "\n",
            "    # 1. VAD cắt khoảng lặng\n",
            "    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)\n",
            "    if len(y_trimmed) < WIN_LEN:\n",
            "        y_trimmed = y\n",
            "\n",
            "    # 2. Tiền nhấn\n",
            "    y_pre = librosa.effects.preemphasis(y_trimmed, coef=PRE_EMPH)\n",
            "\n",
            "    # 3. Trích xuất 13 MFCC tĩnh\n",
            "    mfcc = librosa.feature.mfcc(\n",
            "        y=y_pre, sr=sr, n_mfcc=N_MFCC,\n",
            "        n_fft=N_FFT, win_length=WIN_LEN,\n",
            "        hop_length=HOP_LEN, window='hamming',\n",
            "        n_mels=N_MELS, fmin=20, fmax=sr/2\n",
            "    )\n",
            "\n",
            "    if not use_dynamic:\n",
            "        F = mfcc.T\n",
            "    else:\n",
            "        # 4. Tính toán Delta và Delta-Delta\n",
            "        d1 = librosa.feature.delta(mfcc, order=1)\n",
            "        d2 = librosa.feature.delta(mfcc, order=2)\n",
            "        F = np.vstack([mfcc, d1, d2]).T  # Kích thước T x 39\n",
            "\n",
            "    # 5. Utterance-level CMVN\n",
            "    if apply_cmvn:\n",
            "        mu = F.mean(axis=0, keepdims=True)\n",
            "        sd = F.std(axis=0, keepdims=True) + 1e-8\n",
            "        F = (F - mu) / sd\n",
            "\n",
            "    return F\n",
            "\n",
            "# Kiểm tra shape của 3 file mẫu\n",
            "sample_files = [\n",
            "    sorted(TRAIN_DIR.glob('mo/*.wav'))[0],\n",
            "    sorted(TRAIN_DIR.glob('dong/*.wav'))[0],\n",
            "    sorted(TRAIN_DIR.glob('len/*.wav'))[0]\n",
            "]\n",
            "for f in sample_files:\n",
            "    feat = extract_features(f)\n",
            "    print(f\"File: {f.name:15s} | Feature Matrix Shape: {feat.shape} (T frames x {feat.shape[1]} dims)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Hiển thị Heatmap so sánh đặc trưng 39 chiều của lệnh 'MỞ' và 'ĐÓNG'\n",
            "from IPython.display import Image\n",
            "Image(filename=str(FIGURES_DIR / \"fig1_mfcc39_heatmaps.png\"))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. PHẦN B & C: THIẾT KẾ TOPOLOGY VÀ HUẤN LUYỆN HMM TRÁI-SANG-PHẢI\n",
            "- **Topology Left-to-Right (Bakis Model):** Cấu trúc 5 trạng thái phản ánh tiến trình thời gian của từ nói tiếng Việt (Âm đầu $\\rightarrow$ Chuyển tiếp $\\rightarrow$ Nguyên âm hạt nhân $\\rightarrow$ Âm cuối $\\rightarrow$ Kết thúc).\n",
            "- **Linear Temporal Segmentation:** Khởi tạo $\\mu_i, \\Sigma_i$ bằng cách phân đoạn chuỗi âm thanh thành 5 đoạn đều nhau theo trục thời gian, đảm bảo mô hình không bị suy biến do phân cụm ngẫu nhiên của KMeans.\n",
            "- **Diagonal Gaussian Covariance:** Phù hợp với kích thước tập dữ liệu nhỏ, giảm số tham số hiệp phương sai từ 780 xuống 39 cho mỗi trạng thái."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def train_single_hmm(label, train_files, n_states=5, covariance_type='diag', n_iter=80, use_dynamic=True, random_state=42):\n",
            "    feats = [extract_features(w, use_dynamic=use_dynamic) for w in train_files]\n",
            "    lengths = [len(f) for f in feats]\n",
            "    X = np.vstack(feats)\n",
            "    D = X.shape[1]\n",
            "\n",
            "    # Phân đoạn thời gian đều (Linear Temporal Segmentation)\n",
            "    state_slices = [[] for _ in range(n_states)]\n",
            "    offset = 0\n",
            "    for L in lengths:\n",
            "        seg_len = L / n_states\n",
            "        for s in range(n_states):\n",
            "            st = offset + int(round(s * seg_len))\n",
            "            en = offset + int(round((s + 1) * seg_len))\n",
            "            if en > st:\n",
            "                state_slices[s].append(X[st:en])\n",
            "        offset += L\n",
            "\n",
            "    means = np.array([np.vstack(state_slices[s]).mean(axis=0) for s in range(n_states)])\n",
            "    if covariance_type == 'diag':\n",
            "        covars = np.array([np.vstack(state_slices[s]).var(axis=0) + 1e-2 for s in range(n_states)])\n",
            "    else:\n",
            "        covars = np.array([np.cov(np.vstack(state_slices[s]).T) + np.eye(D) * 1e-2 for s in range(n_states)])\n",
            "\n",
            "    start = np.zeros(n_states)\n",
            "    start[0] = 1.0\n",
            "\n",
            "    A = np.zeros((n_states, n_states))\n",
            "    for i in range(n_states - 1):\n",
            "        A[i, i] = 0.7\n",
            "        A[i, i + 1] = 0.3\n",
            "    A[-1, -1] = 1.0\n",
            "\n",
            "    model = GaussianHMM(\n",
            "        n_components=n_states,\n",
            "        covariance_type=covariance_type,\n",
            "        n_iter=n_iter,\n",
            "        tol=1e-3,\n",
            "        min_covar=1e-3,\n",
            "        init_params='',\n",
            "        params='tmc',\n",
            "        random_state=random_state\n",
            "    )\n",
            "    model.startprob_ = start\n",
            "    model.transmat_ = A\n",
            "    model.means_ = means\n",
            "    model.covars_ = covars\n",
            "\n",
            "    model.fit(X, lengths)\n",
            "\n",
            "    # Bảo toàn cấu trúc Left-to-Right\n",
            "    for i in range(n_states):\n",
            "        for j in range(n_states):\n",
            "            if j < i or j > i + 1:\n",
            "                model.transmat_[i, j] = 0.0\n",
            "        row_sum = np.sum(model.transmat_[i])\n",
            "        if row_sum > 0:\n",
            "            model.transmat_[i] /= row_sum\n",
            "        else:\n",
            "            model.transmat_[i, i] = 1.0\n",
            "\n",
            "    return model\n",
            "\n",
            "# Huấn luyện 5 mô hình baseline\n",
            "models = {}\n",
            "for label in LABELS:\n",
            "    files = sorted(TRAIN_DIR.glob(f\"{label}/*.wav\"))\n",
            "    models[label] = train_single_hmm(label, files, n_states=5, covariance_type='diag', use_dynamic=True)\n",
            "    print(f\"Huấn luyện thành công Model [{label.upper():6s}] | Converged: {models[label].monitor_.converged} ({len(models[label].monitor_.history)} EM iters)\")\n",
            "    print(f\"Ma trận chuyển trạng thái sau fit (Transmat):\\n{np.round(models[label].transmat_, 3)}\\n\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. PHẦN D: NHẬN DẠNG MAXIMUM LIKELIHOOD & CĂN CHỈNH VITERBI\n",
            "- **Nguyên tắc quyết định:** $\\hat{w}_{ML} = \\arg\\max_w \\log P(X | \\lambda_w)$ bằng thuật toán Forward.\n",
            "- **Viterbi State Alignment:** Tìm chuỗi trạng thái tối ưu $Q^*$ và trực quan hóa Trellis step plot."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def recognize(feat, models):\n",
            "    scores = {lbl: models[lbl].score(feat) for lbl in models}\n",
            "    pred = max(scores, key=scores.get)\n",
            "    return pred, scores\n",
            "\n",
            "# Đánh giá tập Test sạch\n",
            "true_y, pred_y = [], []\n",
            "all_scores = []\n",
            "for label in LABELS:\n",
            "    for wav in sorted((TEST_DIR / label).glob('*.wav')):\n",
            "        feat = extract_features(wav)\n",
            "        pred, scores = recognize(feat, models)\n",
            "        true_y.append(label)\n",
            "        pred_y.append(pred)\n",
            "        all_scores.append({'file': wav.name, 'true': label, 'pred': pred, 'scores': scores})\n",
            "\n",
            "acc = accuracy_score(true_y, pred_y)\n",
            "print(f\"=== ĐỘ CHÍNH XÁC BASELINE HMM TRÊN TẬP TEST SẠCH: {acc * 100:.2f}% ({sum(np.array(true_y)==np.array(pred_y))}/{len(true_y)}) ===\\n\")\n",
            "\n",
            "# Hiển thị bảng điểm Acoustic Score cho 3 file kiểm thử\n",
            "print(f\"{'Tệp kiểm thử':<16} | {'Nhãn gốc':<8} | {'Dự đoán':<8} | \" + \" | \".join([f\"{l.upper():<8}\" for l in LABELS]))\n",
            "print(\"-\" * 75)\n",
            "for s in all_scores[:3]:\n",
            "    s_str = \" | \".join([f\"{s['scores'][l]:8.1f}\" for l in LABELS])\n",
            "    print(f\"{s['file']:<16} | {s['true']:<8} | {s['pred']:<8} | {s_str}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Hiển thị Confusion Matrix và Biểu đồ Căn chỉnh trạng thái Viterbi\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "cm_img = plt.imread(str(FIGURES_DIR / \"fig4_confusion_matrix_hmm.png\"))\n",
            "vit_img = plt.imread(str(FIGURES_DIR / \"fig2_viterbi_alignment_mo.png\"))\n",
            "axes[0].imshow(cm_img); axes[0].axis('off'); axes[0].set_title(\"Ma trận nhầm lẫn Baseline HMM\", fontweight='bold')\n",
            "axes[1].imshow(vit_img); axes[1].axis('off'); axes[1].set_title(\"Viterbi Alignment: mo_07.wav vs HMM 'MO'\", fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. PHẦN E: ĐỐI CHUẨN THỰC NGHIỆM VỚI DTW BASELINE (LAB 2)\n",
            "Bảng so sánh đối chuẩn giữa thuật toán DTW (Dynamic Time Warping) và Gaussian HMM trên cùng tập dữ liệu:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image\n",
            "Image(filename=str(FIGURES_DIR / \"fig5_dtw_vs_hmm_comparison.png\"))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "| Chỉ số kỹ thuật | DTW Baseline (Lab 2) | Gaussian HMM (Lab 3) | Nhận xét chuyên sâu |\n",
            "| :--- | :---: | :---: | :--- |\n",
            "| **Bản chất mô hình** | So khớp mẫu trực tiếp | Mô hình sinh thống kê | HMM học phân bố xác suất từ nhiều mẫu |\n",
            "| **Accuracy (Test sạch)** | 100.0% | 100.0% | Cả hai hoạt động hoàn hảo với từ đơn tách rời |\n",
            "| **Số mẫu train yêu cầu** | 1 - 2 mẫu | $\\ge 6$ mẫu | DTW chỉ cần 1 template; HMM cần đủ dữ liệu để ước lượng $\\mu, \\Sigma$ |\n",
            "| **Thời gian huấn luyện** | $0.00\\text{ s}$ | $\\approx 0.37\\text{ s}$ | DTW chỉ lưu vector; HMM cần tối ưu hóa Baum-Welch (EM) |\n",
            "| **Độ trễ nhận dạng (Test)** | Tăng tuyến tính theo số template | Cố định theo số mô hình lớp từ | HMM xử lý cực nhanh ở pha inference |\n",
            "| **Cơ chế căn chỉnh thời gian** | Warping path $(i, j)$ | Chuỗi trạng thái Viterbi $q_t^*$ | DTW so mẫu-mẫu; HMM so mẫu-trạng thái trừu tượng |\n",
            "| **Khả năng mở rộng từ vựng** | Kém (Bùng nổ tính toán) | Rất cao (Tích hợp Lexicon, LM) | HMM là nền tảng của hệ nhận dạng tiếng nói lớn |"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. PHẦN F & G: PRONUNCIATION LEXICON VÀ BIGRAM LANGUAGE MODEL RE-RANKING\n",
            "Nguyên lý ASR theo Bayes tích hợp Acoustic Model và Language Model:\n",
            "$$\\text{Score}(W) = \\frac{\\log P_A(X|W)}{T} + \\lambda_{LM} \\log P_L(W | W_{\\text{prev}})$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Đọc và giải thích Lexicon\n",
            "def parse_lexicon(filepath):\n",
            "    lex = {}\n",
            "    with open(filepath, 'r', encoding='utf-8') as f:\n",
            "        for line in f:\n",
            "            if line.strip() and not line.startswith('#'):\n",
            "                parts = line.strip().split()\n",
            "                lex[parts[0]] = parts[1:]\n",
            "    return lex\n",
            "\n",
            "lexicon = parse_lexicon(BASE_DIR / \"lexicon.txt\")\n",
            "print(\"=== BẢNG PHIÊN ÂM TIẾNG VIỆT (PRONUNCIATION LEXICON) ===\")\n",
            "for word, phones in lexicon.items():\n",
            "    print(f\"  Lệnh: {word:6s} -> Chuỗi âm vị: {' - '.join(phones)}\")\n",
            "\n",
            "# 2. Hiển thị đồ thị Re-ranking theo lambda_LM\n",
            "Image(filename=str(FIGURES_DIR / \"fig6_lm_reranking_analysis.png\"))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. PHẦN H: BỘ 6 THÍ NGHIỆM BẮT BUỘC (E1 $\\rightarrow$ E6) & CƠ CHẾ REJECT THRESHOLD\n",
            "Các thí nghiệm được thực hiện có kiểm soát biến độc lập theo chuẩn đề bài:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "df_results = pd.read_csv(BASE_DIR / \"results.csv\")\n",
            "print(\"=== BẢNG TỔNG HỢP SỐ LIỆU ĐỊNH LƯỢNG 6 THÍ NGHIỆM (E1 -> E6) ===\")\n",
            "display(df_results)\n",
            "\n",
            "# Hiển thị đồ thị phân bố điểm số và ngưỡng từ chối Reject Threshold\n",
            "Image(filename=str(FIGURES_DIR / \"fig7_score_distributions_reject.png\"))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Nhận xét kết quả 6 thí nghiệm:\n",
            "1. **E1 (Số trạng thái):** Với các từ đơn tiết tiếng Việt, $N=5$ trạng thái là cấu hình tối ưu nhất về khả năng biểu diễn ngữ âm theo thời gian mà vẫn đảm bảo số lượng frame trung bình mỗi trạng thái đủ lớn để ước lượng ma trận hiệp phương sai ổn định.\n",
            "2. **E2 (Đặc trưng động):** Bổ sung $\\Delta$ và $\\Delta^2$ mang lại thông tin vận tốc và gia tốc thay đổi phổ, giúp mô hình vượt qua giả thiết độc lập quan sát (output independence) của HMM chuẩn.\n",
            "3. **E3 (Hiệp phương sai):** Với tập dữ liệu nhỏ (6 mẫu/từ), `covariance_type=\"diag\"` là bắt buộc vì ma trận `full` đòi hỏi ước lượng 780 tham số/state dẫn đến nguy cơ suy biến (singularity) và overfit cao.\n",
            "4. **E4 (Số mẫu huấn luyện):** DTW hoạt động vượt trội khi số mẫu rất ít ($K=1$), nhưng HMM đạt độ bền vững cao hơn khi có đủ mẫu huấn luyện.\n",
            "5. **E5 (Mismatch):** Hệ thống duy trì độ chính xác cao khi kiểm thử chéo người nói (Speaker 2), nhưng sụt giảm xuống $40\\%$ khi có nhiễu nền mạnh (SNR 15dB), cho thấy sự cần thiết của các kỹ thuật khử nhiễu tiền xử lý.\n",
            "6. **E6 (Trọng số LM):** $\\lambda_{LM} = 0.3$ là giá trị hài hòa giúp hỗ trợ giải mã trong các trường hợp âm học không chắc chắn mà không lấn át bằng chứng âm học thực tế.\n",
            "7. **Cơ chế Reject Threshold:** Ngưỡng điểm Normalized Log-likelihood $\\theta_{\\text{score}} = -46.20$ cho phép từ chối các từ ngoài từ vựng (Out-of-Vocabulary) một cách tin cậy."
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
    "nbformat_minor": 4
}

out_nb_path = BASE_DIR / "Lab03.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Đã tạo thành công Notebook: {out_nb_path.name}")
