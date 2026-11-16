"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
Mô-đun 2: Nhận Dạng & Xác Thực Người Nói (Speaker Recognition)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Nội dung:
- Trích xuất 20 MFCC + 20 Delta MFCC, hỗ trợ tùy chọn Cepstral Mean Normalization (CMN).
- Nén thống kê (mean/std) thành vector embedding 80 chiều hoặc 40 chiều (ablation).
- Chuẩn hóa L2-norm và mô hình hóa Centroid người nói từ các câu enrollment.
- Speaker Identification: Cosine similarity, Confusion matrix, Score margin.
- Speaker Verification: Phân tách cặp Genuine/Impostor, quét ngưỡng theta trên tập Dev
  tìm điểm cân bằng Equal Error Rate (EER), sau đó khóa ngưỡng đánh giá trên tập Test độc lập.
===============================================================================
"""

import numpy as np
import librosa
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix


def extract_mfcc_features(y, sr=16000, n_mfcc=20, n_fft=512, win_length=400, hop_length=160, use_cmn=False):
    """
    Trích xuất ma trận MFCC và Delta MFCC trên tín hiệu y.
    Có tùy chọn Cepstral Mean Normalization (CMN): c_hat_t = c_t - c_bar.
    """
    M = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        window='hann'
    )
    if use_cmn:
        M = M - np.mean(M, axis=1, keepdims=True)

    D = librosa.feature.delta(M)
    return M, D


def extract_speaker_embedding(y, sr=16000, n_mfcc=20, include_delta=True, use_cmn=False):
    """
    Tạo vector embedding thống kê thời gian:
    - Nếu include_delta=True: e = [mean(M), std(M), mean(D), std(D)] (80 chiều).
    - Nếu include_delta=False: e = [mean(M), std(M)] (40 chiều) cho thí nghiệm ablation.
    Vector được chuẩn hóa L2 norm để ổn định số học.
    """
    M, D = extract_mfcc_features(y, sr=sr, n_mfcc=n_mfcc, use_cmn=use_cmn)

    if include_delta:
        e = np.concatenate([
            np.mean(M, axis=1),
            np.std(M, axis=1),
            np.mean(D, axis=1),
            np.std(D, axis=1)
        ])
    else:
        e = np.concatenate([
            np.mean(M, axis=1),
            np.std(M, axis=1)
        ])

    e_norm = norm(e) + 1e-12
    return (e / e_norm).astype(np.float64)


def cosine_similarity(a, b):
    """Tính độ tương đồng Cosine giữa 2 vector a và b."""
    return float(np.dot(a, b) / ((norm(a) * norm(b)) + 1e-12))


def train_speaker_models(enrollment_dict):
    """
    Huấn luyện mô hình người nói từ danh sách các vector enrollment.
    Mỗi speaker có vector model m_i là centroid (trung bình) của các embedding,
    sau đó được chuẩn hóa L2 norm.
    enrollment_dict: { 'spk01': [e1, e2, e3, e4], 'spk02': [...], ... }
    """
    models = {}
    for spk, embs in enrollment_dict.items():
        centroid = np.mean(embs, axis=0)
        models[spk] = (centroid / (norm(centroid) + 1e-12)).astype(np.float64)
    return models


def identify_utterance(test_emb, speaker_models):
    """
    Dự đoán danh tính người nói cho 1 câu test bằng Cosine similarity.
    Trả về: predicted_spk, scores_dict, margin (khoảng cách top-1 với top-2).
    """
    scores = {spk: cosine_similarity(test_emb, m) for spk, m in speaker_models.items()}
    sorted_speakers = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_spk, best_score = sorted_speakers[0]
    second_score = sorted_speakers[1][1] if len(sorted_speakers) > 1 else -1.0
    margin = best_score - second_score
    return best_spk, scores, float(margin)


def evaluate_identification(test_dict, speaker_models):
    """
    Đánh giá Speaker Identification trên toàn bộ tập test:
    test_dict: { 'spk01': [e_test1, e_test2], ... }
    Trả về: accuracy, confusion_mat, speakers_list, detailed_results
    """
    speakers = sorted(list(speaker_models.keys()))
    spk2idx = {s: i for i, s in enumerate(speakers)}

    y_true = []
    y_pred = []
    detailed_results = []
    margins = []

    for true_spk, embs in test_dict.items():
        for i, emb in enumerate(embs):
            pred_spk, scores, margin = identify_utterance(emb, speaker_models)
            y_true.append(true_spk)
            y_pred.append(pred_spk)
            margins.append(margin)
            detailed_results.append({
                "true_speaker": true_spk,
                "utterance_idx": i,
                "predicted_speaker": pred_spk,
                "correct": (true_spk == pred_spk),
                "margin": margin,
                "scores": scores
            })

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    cm = confusion_matrix(
        [spk2idx[t] for t in y_true],
        [spk2idx[p] for p in y_pred],
        labels=list(range(len(speakers)))
    )
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "speakers": speakers,
        "avg_margin": float(np.mean(margins)),
        "detailed_results": detailed_results
    }


def build_verification_trials(test_embeddings_dict, speaker_models):
    """
    Tạo các cặp thử nghiệm xác thực:
    - Genuine trial (label=1): test câu của spk_i so sánh với model của spk_i.
    - Impostor trial (label=0): test câu của spk_i so sánh với model của spk_j (j != i).
    """
    trials = []
    for true_spk, embs in test_embeddings_dict.items():
        for emb in embs:
            for model_spk, m in speaker_models.items():
                score = cosine_similarity(emb, m)
                label = 1 if (true_spk == model_spk) else 0
                trials.append((float(score), label, true_spk, model_spk))
    return trials


def sweep_thresholds(dev_trials, n_points=401):
    """
    Quét ngưỡng theta trong khoảng [-1.0, 1.0] trên tập Development.
    Tính FAR(theta) và FRR(theta) tại mỗi ngưỡng, tìm ngưỡng theta* tối ưu tại điểm EER.
    """
    thresholds = np.linspace(-1.0, 1.0, n_points)
    genuine_scores = np.array([t[0] for t in dev_trials if t[1] == 1])
    impostor_scores = np.array([t[0] for t in dev_trials if t[1] == 0])

    curve_data = []
    for th in thresholds:
        frr = float(np.mean(genuine_scores < th))
        far = float(np.mean(impostor_scores >= th))
        curve_data.append((th, far, frr, abs(far - frr)))

    # Điểm tối ưu gần EER nhất
    best_point = min(curve_data, key=lambda z: z[3])
    theta_eer = best_point[0]
    eer_approx = (best_point[1] + best_point[2]) / 2.0

    return {
        "thresholds": thresholds,
        "curve_data": curve_data,
        "theta_eer": float(theta_eer),
        "eer": float(eer_approx),
        "far_at_eer": float(best_point[1]),
        "frr_at_eer": float(best_point[2])
    }


def evaluate_verification(test_trials, threshold):
    """Đánh giá FAR và FRR trên tập Test độc lập với ngưỡng theta đã được khóa."""
    genuine_scores = np.array([t[0] for t in test_trials if t[1] == 1])
    impostor_scores = np.array([t[0] for t in test_trials if t[1] == 0])

    frr = float(np.mean(genuine_scores < threshold))
    far = float(np.mean(impostor_scores >= threshold))
    return {
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores)
    }
