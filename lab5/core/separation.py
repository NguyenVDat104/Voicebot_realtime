"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
Mô-đun 3: Phân Tách Nguồn Nói Bằng FastICA (Blind Source Separation)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Nội dung:
- Mô hình trộn tức thời x = A * s với 2 nguồn nói và 2 micro.
- Đánh giá định thức det(A) và giải thích tính khả nghịch của ma trận trộn.
- Áp dụng giải thuật FastICA (Scikit-learn) để ước lượng các thành phần độc lập.
- Khử 2 bất định cốt lõi của ICA:
    (1) Permutation ambiguity: ghép nối bằng tương quan tuyệt đối |rho|.
    (2) Sign ambiguity: đảo dấu nếu tương quan âm (rho < 0).
- Chuẩn hóa biên độ peak trước khi lưu file WAV.
- Thử nghiệm ma trận trộn điều kiện xấu và hỗn hợp có cộng thêm nhiễu.
===============================================================================
"""

import numpy as np
from sklearn.decomposition import FastICA


def rms_normalize(x):
    """Chuẩn hóa tín hiệu về đơn vị RMS để 2 nguồn cân bằng công suất."""
    x = np.asarray(x, dtype=np.float64)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def peak_normalize(x, target_peak=0.90):
    """Chuẩn hóa biên độ đỉnh (peak) để không bị clip âm thanh khi lưu WAV."""
    peak = np.max(np.abs(x))
    if peak > 0:
        return (x / peak) * target_peak
    return x


def create_instantaneous_mixture(s1, s2, A=None):
    """
    Tạo tín hiệu hỗn hợp quan sát tại 2 microphone:
    X = A @ S, trong đó S = [s1, s2]^T có shape (2, L).
    """
    if A is None:
        A = np.array([
            [1.0, 0.6],
            [0.4, 1.0]
        ], dtype=np.float64)
    else:
        A = np.asarray(A, dtype=np.float64)

    L = min(len(s1), len(s2))
    s1_norm = rms_normalize(s1[:L])
    s2_norm = rms_normalize(s2[:L])
    S = np.vstack([s1_norm, s2_norm])  # (2, L)

    X = A @ S  # (2, L)
    det_A = float(np.linalg.det(A))
    return S, X, A, det_A


def run_fastica_separation(X, random_state=0):
    """
    Chạy thuật toán FastICA với 2 thành phần độc lập.
    Đầu vào X có shape (2, L), chuyển vị thành (L, 2) cho scikit-learn.
    Đầu ra Shat có shape (2, L).
    """
    ica = FastICA(n_components=2, whiten='unit-variance', random_state=random_state)
    Shat = ica.fit_transform(X.T).T  # Shape: (2, L)
    return Shat, ica


def compute_correlation_matrix(S, Shat):
    """
    Tính ma trận hệ số tương quan Pearson giữa nguồn gốc S và nguồn ước lượng Shat:
    C_raw[i, j] = corrcoef(S[i], Shat[j])
    C_abs[i, j] = |corrcoef(S[i], Shat[j])|
    """
    C_raw = np.zeros((2, 2), dtype=np.float64)
    C_abs = np.zeros((2, 2), dtype=np.float64)

    for i in range(2):
        for j in range(2):
            val = np.corrcoef(S[i], Shat[j])[0, 1]
            C_raw[i, j] = val
            C_abs[i, j] = abs(val)

    return C_raw, C_abs


def resolve_ica_ambiguities(S, Shat):
    """
    Giải quyết triệt để 2 bất định của ICA:
    1. Permutation Ambiguity: Ghép estimate với source dựa trên tương quan tuyệt đối lớn nhất.
    2. Sign Ambiguity: Nếu tương quan mang dấu âm, nhân -1 vào estimate.
    Trả về Shat_resolved (đã ghép đúng thứ tự và cùng dấu) cùng ma trận tương quan đã hiệu chỉnh.
    """
    C_raw, C_abs = compute_correlation_matrix(S, Shat)

    # Kiểm tra phương án ghép cặp tối ưu (0->0, 1->1) hay (0->1, 1->0)
    score_direct = C_abs[0, 0] + C_abs[1, 1]
    score_permuted = C_abs[0, 1] + C_abs[1, 0]

    if score_direct >= score_permuted:
        perm = [0, 1]
    else:
        perm = [1, 0]

    Shat_resolved = np.zeros_like(Shat)
    aligned_correlations = []

    for src_idx in range(2):
        est_idx = perm[src_idx]
        raw_corr = C_raw[src_idx, est_idx]
        est_signal = Shat[est_idx].copy()

        # Khử đảo dấu (Sign ambiguity)
        if raw_corr < 0:
            est_signal = -est_signal
            corr_final = -raw_corr
        else:
            corr_final = raw_corr

        Shat_resolved[src_idx] = peak_normalize(est_signal, 0.90)
        aligned_correlations.append(float(corr_final))

    C_raw_final, C_abs_final = compute_correlation_matrix(S, Shat_resolved)

    return {
        "Shat_resolved": Shat_resolved,
        "permutation": perm,
        "aligned_correlations": aligned_correlations,
        "C_abs_before": C_abs,
        "C_abs_after": C_abs_final
    }


def separate_sources_pipeline(s1, s2, A=None, random_state=0, add_noise_std=0.0):
    """
    Đường ống phân tách nguồn toàn diện:
    1. Tạo hỗn hợp trộn tại 2 mic (có tùy chọn thêm nhiễu trắng nhẹ).
    2. Chạy FastICA.
    3. Giải quyết permutation và sign.
    4. Trả về tín hiệu đã tách và báo cáo định lượng.
    """
    S, X, A_mat, det_A = create_instantaneous_mixture(s1, s2, A=A)

    if add_noise_std > 0.0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, add_noise_std, X.shape)
        X = X + noise

    Shat, ica = run_fastica_separation(X, random_state=random_state)
    res = resolve_ica_ambiguities(S, Shat)

    res["S_ground_truth"] = S
    res["X_mixture"] = X
    res["A"] = A_mat
    res["det_A"] = det_A
    return res
