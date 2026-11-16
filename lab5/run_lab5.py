"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
Kịch Bản Thực Thi Toàn Bộ Thực Nghiệm & Tích Hợp (Master Experiment Runner)
Học phần: CSE457 - Trường Đại học Thủy Lợi
Sinh viên: Nguyễn Văn Đạt

Quy trình:
1. Nạp dữ liệu 16kHz từ lab5_data/
2. Thực thi Case Study: "Trợ lý phòng họp mini" (5 bước)
3. Thực thi 8 Thí nghiệm bắt buộc (TN1 -> TN8)
4. Lưu toàn bộ file âm thanh đầu ra vào results/audio/
5. Lưu toàn bộ biểu đồ khoa học vào results/figures/
6. Xuất bảng tổng hợp kết quả định lượng vào results/metrics_summary.csv
===============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Thêm đường dẫn thư mục lab5 vào sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.enhancement import (
    load16k, power, rms, add_noise_at_snr, snr_db,
    enhance_spectral_subtraction, enhance_wiener, enhance_both,
    stft_forward
)
from core.speaker_rec import (
    extract_speaker_embedding, train_speaker_models,
    identify_utterance, evaluate_identification,
    build_verification_trials, sweep_thresholds, evaluate_verification
)
from core.separation import (
    create_instantaneous_mixture, run_fastica_separation,
    resolve_ica_ambiguities, separate_sources_pipeline
)

# Cấu hình đường dẫn
DATA_DIR = BASE_DIR / "lab5_data"
RESULTS_DIR = BASE_DIR / "results"
AUDIO_OUT_DIR = RESULTS_DIR / "audio"
FIGS_OUT_DIR = RESULTS_DIR / "figures"

AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình font chữ Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_spectrogram(y, sr, title, ax):
    """Vẽ biểu đồ spectrogram tần số logarit."""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, hop_length=160, ax=ax, cmap='magma')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Thời gian (s)', fontsize=9)
    ax.set_ylabel('Tần số (Hz)', fontsize=9)
    return img


def run_case_study():
    print("\n" + "=" * 75)
    print(">>> GIAI ĐOẠN 1: CASE STUDY - 'TRỢ LÝ PHÒNG HỌP MINI' (MỤC 5 LAB 5)")
    print("=" * 75)

    # 1. Bước 1: Nạp audio & tạo noisy speech tại SNR = 5 dB
    clean_path = DATA_DIR / "clean" / "spk01_test_01.wav"
    noise_path = DATA_DIR / "noise" / "fan.wav"

    s, sr = load16k(str(clean_path))
    v, _ = load16k(str(noise_path))

    target_snr = 5.0
    y_noisy, alpha = add_noise_at_snr(s, v, target_snr)
    measured_snr_in = snr_db(s, y_noisy)

    noisy_wav_path = AUDIO_OUT_DIR / "noisy_5dB.wav"
    sf.write(noisy_wav_path, y_noisy.astype(np.float32), sr, subtype="PCM_16")

    print(f"[Bước 1] Đã tạo Noisy Speech (spk01 + fan noise):")
    print(f"         - Target SNR: {target_snr:.1f} dB | Measured SNR: {measured_snr_in:.2f} dB (Hệ số alpha = {alpha:.4f})")
    print(f"         - Lưu file: {noisy_wav_path.relative_to(BASE_DIR)}")

    # 2. Bước 2: Khử nhiễu STFT bằng Spectral Subtraction và Wiener
    x_ss = enhance_spectral_subtraction(y_noisy, v, sr=sr, beta=1.5, floor=0.02)
    x_w, H = enhance_wiener(y_noisy, v, sr=sr)

    snr_out_ss = snr_db(s, x_ss)
    snr_out_w = snr_db(s, x_w)
    delta_ss = snr_out_ss - measured_snr_in
    delta_w = snr_out_w - measured_snr_in

    ss_wav_path = AUDIO_OUT_DIR / "enhanced_ss.wav"
    w_wav_path = AUDIO_OUT_DIR / "enhanced_wiener.wav"
    sf.write(ss_wav_path, x_ss.astype(np.float32), sr, subtype="PCM_16")
    sf.write(w_wav_path, x_w.astype(np.float32), sr, subtype="PCM_16")

    print(f"\n[Bước 2] Kết quả Khử nhiễu STFT:")
    print(f"         - Spectral Subtraction: SNR_out = {snr_out_ss:.2f} dB | Delta_SNR = +{delta_ss:.2f} dB")
    print(f"         - Wiener Filter:        SNR_out = {snr_out_w:.2f} dB | Delta_SNR = +{delta_w:.2f} dB")
    print(f"         - Lưu file: {ss_wav_path.relative_to(BASE_DIR)}, {w_wav_path.relative_to(BASE_DIR)}")

    # Vẽ biểu đồ Spectrogram 4 ô: Clean vs Noisy vs SS vs Wiener
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    plot_spectrogram(s, sr, "1. Clean Reference (spk01)", axs[0, 0])
    plot_spectrogram(y_noisy, sr, f"2. Noisy (SNR = {measured_snr_in:.2f} dB)", axs[0, 1])
    plot_spectrogram(x_ss, sr, f"3. Spectral Subtraction (SNR = {snr_out_ss:.2f} dB, Delta = +{delta_ss:.2f} dB)", axs[1, 0])
    img_last = plot_spectrogram(x_w, sr, f"4. Wiener Filter (SNR = {snr_out_w:.2f} dB, Delta = +{delta_w:.2f} dB)", axs[1, 1])
    fig.colorbar(img_last, ax=axs, format="%+2.0f dB", fraction=0.02, pad=0.04)
    fig.suptitle("So Sánh Phổ Spectrogram Trước Và Sau Khử Nhiễu (STFT Domain)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1_path = FIGS_OUT_DIR / "fig1_enhancement_spectrograms.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"         - Đã lưu biểu đồ phổ: {fig1_path.relative_to(BASE_DIR)}")

    # 3. Bước 3: Speaker Embedding & Speaker Identification
    speakers = ["spk01", "spk02", "spk03"]
    enrollment_data = {spk: [] for spk in speakers}
    for spk in speakers:
        for i in range(1, 5):
            path = DATA_DIR / "clean" / f"{spk}_enroll_{i:02d}.wav"
            audio, _ = load16k(str(path))
            emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=False)
            enrollment_data[spk].append(emb)

    speaker_models = train_speaker_models(enrollment_data)
    print(f"\n[Bước 3] Đã huấn luyện Centroid Models cho 3 người nói ({list(speaker_models.keys())})")

    # Thử nghiệm câu test spk01 trên 4 biến thể: Clean, Noisy, SS, Wiener
    eval_variants = {
        "Clean": s,
        "Noisy (5dB)": y_noisy,
        "Enhanced SS": x_ss,
        "Enhanced Wiener": x_w
    }
    print("         >> Điểm số nhận dạng câu spk01_test_01:")
    case_study_id_scores = []
    for var_name, var_audio in eval_variants.items():
        emb = extract_speaker_embedding(var_audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=False)
        pred, scores, margin = identify_utterance(emb, speaker_models)
        case_study_id_scores.append({
            "Variant": var_name,
            "Pred": pred,
            "Score_spk01": scores["spk01"],
            "Score_spk02": scores["spk02"],
            "Score_spk03": scores["spk03"],
            "Margin": margin
        })
        print(f"            * {var_name:16s} -> Dự đoán: {pred} | Score(spk01) = {scores['spk01']:.4f} | Margin = {margin:.4f}")

    # Đánh giá Identification trên toàn bộ tập test
    test_data = {spk: [] for spk in speakers}
    for spk in speakers:
        for i in range(1, 3):
            path = DATA_DIR / "clean" / f"{spk}_test_{i:02d}.wav"
            audio, _ = load16k(str(path))
            emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=False)
            test_data[spk].append(emb)

    id_eval = evaluate_identification(test_data, speaker_models)
    print(f"         >> Độ chính xác Identification trên tập Test Clean: {id_eval['accuracy']*100:.1f}%")

    # Vẽ Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = id_eval["confusion_matrix"]
    cax = ax.matshow(cm, cmap='Blues')
    for i in range(len(speakers)):
        for j in range(len(speakers)):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > np.max(cm)/2 else 'black')
    fig.colorbar(cax)
    ax.set_xticks(range(len(speakers)))
    ax.set_yticks(range(len(speakers)))
    ax.set_xticklabels(speakers, fontsize=11)
    ax.set_yticklabels(speakers, fontsize=11)
    ax.set_xlabel('Predicted Speaker', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Speaker', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Speaker Identification (Test Set)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig3_path = FIGS_OUT_DIR / "fig3_confusion_matrix.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"         - Đã lưu biểu đồ: {fig3_path.relative_to(BASE_DIR)}")

    # 4. Bước 4: Speaker Verification Threshold & FAR / FRR
    # Chia enrollment thành: 3 câu huấn luyện, 1 câu làm Development set để quét ngưỡng theta
    dev_enroll_dict = {spk: enrollment_data[spk][:3] for spk in speakers}
    dev_models = train_speaker_models(dev_enroll_dict)
    dev_test_dict = {spk: [enrollment_data[spk][3]] for spk in speakers}  # Câu enroll thứ 4 làm dev

    dev_trials = build_verification_trials(dev_test_dict, dev_models)
    sweep_res = sweep_thresholds(dev_trials, n_points=401)
    theta_star = sweep_res["theta_eer"]

    print(f"\n[Bước 4] Phân tích Verification Threshold:")
    print(f"         - Ngưỡng tối ưu xác định trên tập Dev: theta* = {theta_star:.4f}")
    print(f"         - Tại điểm cân bằng EER (Dev): FAR = {sweep_res['far_at_eer']*100:.2f}%, FRR = {sweep_res['frr_at_eer']*100:.2f}%")

    # Khóa ngưỡng theta* và đánh giá trên Test set độc lập
    test_trials = build_verification_trials(test_data, dev_models)
    test_verif_res = evaluate_verification(test_trials, threshold=theta_star)
    print(f"         - Kết quả kiểm thử độc lập (Test Set, locked theta*={theta_star:.4f}):")
    print(f"           FAR(test) = {test_verif_res['far']*100:.2f}% | FRR(test) = {test_verif_res['frr']*100:.2f}%")

    # Vẽ đường cong FAR - FRR theo ngưỡng theta
    fig, ax = plt.subplots(figsize=(8, 5))
    th_vals = sweep_res["thresholds"]
    far_vals = [pt[1] for pt in sweep_res["curve_data"]]
    frr_vals = [pt[2] for pt in sweep_res["curve_data"]]

    ax.plot(th_vals, far_vals, label='FAR (False Accept Rate)', color='crimson', lw=2)
    ax.plot(th_vals, frr_vals, label='FRR (False Reject Rate)', color='navy', lw=2)
    ax.axvline(theta_star, color='darkgreen', linestyle='--', label=f'Optimal Theta* = {theta_star:.3f} (EER)')
    ax.scatter([theta_star], [sweep_res['eer']], color='darkgreen', s=70, zorder=5)
    ax.set_title('Đường Cong Trade-Off FAR - FRR Theo Ngưỡng Theta (Development Set)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ngưỡng Xác Thực Theta (Cosine Score)', fontsize=11)
    ax.set_ylabel('Tỷ lệ Lỗi (Error Rate)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig4_path = FIGS_OUT_DIR / "fig4_far_frr_curve.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"         - Đã lưu biểu đồ: {fig4_path.relative_to(BASE_DIR)}")

    # 5. Bước 5: Phân tách nguồn BSS / FastICA
    srcA_path = DATA_DIR / "separation" / "source_A.wav"
    srcB_path = DATA_DIR / "separation" / "source_B.wav"
    s1, _ = load16k(str(srcA_path))
    s2, _ = load16k(str(srcB_path))

    A_standard = np.array([[1.0, 0.6], [0.4, 1.0]])
    sep_res = separate_sources_pipeline(s1, s2, A=A_standard, random_state=0)

    # Lưu 2 mixture và 2 separated WAV
    mix1_wav = AUDIO_OUT_DIR / "mix_mic1.wav"
    mix2_wav = AUDIO_OUT_DIR / "mix_mic2.wav"
    sep1_wav = AUDIO_OUT_DIR / "separated_1.wav"
    sep2_wav = AUDIO_OUT_DIR / "separated_2.wav"

    sf.write(mix1_wav, sep_res["X_mixture"][0].astype(np.float32), sr, subtype="PCM_16")
    sf.write(mix2_wav, sep_res["X_mixture"][1].astype(np.float32), sr, subtype="PCM_16")
    sf.write(sep1_wav, sep_res["Shat_resolved"][0].astype(np.float32), sr, subtype="PCM_16")
    sf.write(sep2_wav, sep_res["Shat_resolved"][1].astype(np.float32), sr, subtype="PCM_16")

    print(f"\n[Bước 5] Kết quả Phân tách Nguồn FastICA:")
    print(f"         - Ma trận trộn A:\n{A_standard} (det(A) = {sep_res['det_A']:.4f})")
    print(f"         - Ma trận |Correlation| Trước ghép:\n{sep_res['C_abs_before']}")
    print(f"         - Thứ tự hoán vị phát hiện: {sep_res['permutation']}")
    print(f"         - Tương quan thẳng hàng sau khi khử Permutation & Sign:")
    print(f"           Source A <-> Estimate 1: {sep_res['aligned_correlations'][0]:.4f}")
    print(f"           Source B <-> Estimate 2: {sep_res['aligned_correlations'][1]:.4f}")
    print(f"         - Lưu file: mix_mic1.wav, mix_mic2.wav, separated_1.wav, separated_2.wav")

    # Vẽ biểu đồ tương quan ICA và dạng sóng tách
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    im1 = axs[0].imshow(sep_res['C_abs_before'], cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title('1. |Correlation| Trước khi ghép', fontsize=11, fontweight='bold')
    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[0].set_xticklabels(['Est 0', 'Est 1'])
    axs[0].set_yticklabels(['Src A', 'Src B'])
    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, f"{sep_res['C_abs_before'][i, j]:.3f}", ha='center', va='center',
                        color='white' if sep_res['C_abs_before'][i, j] < 0.6 else 'black', fontweight='bold')

    im2 = axs[1].imshow(sep_res['C_abs_after'], cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title('2. |Correlation| Sau khi giải Permutation & Sign', fontsize=11, fontweight='bold')
    axs[1].set_xticks([0, 1])
    axs[1].set_yticks([0, 1])
    axs[1].set_xticklabels(['Separated 1', 'Separated 2'])
    axs[1].set_yticklabels(['Src A', 'Src B'])
    for i in range(2):
        for j in range(2):
            axs[1].text(j, i, f"{sep_res['C_abs_after'][i, j]:.3f}", ha='center', va='center',
                        color='white' if sep_res['C_abs_after'][i, j] < 0.6 else 'black', fontweight='bold')

    fig.colorbar(im2, ax=axs)
    plt.tight_layout()
    fig5_path = FIGS_OUT_DIR / "fig5_ica_correlation.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"         - Đã lưu biểu đồ: {fig5_path.relative_to(BASE_DIR)}")

    return {
        "case_study_id": case_study_id_scores,
        "delta_ss": delta_ss,
        "delta_w": delta_w,
        "theta_star": theta_star,
        "verif_test_res": test_verif_res,
        "ica_corrs": sep_res["aligned_correlations"]
    }


def run_mandatory_experiments():
    print("\n" + "=" * 75)
    print(">>> GIAI ĐOẠN 2: THỰC THI 8 THÍ NGHIỆM BẮT BUỘC (TN1 -> TN8 - MỤC 6 LAB 5)")
    print("=" * 75)

    clean_path = DATA_DIR / "clean" / "spk01_test_01.wav"
    fan_path = DATA_DIR / "noise" / "fan.wav"
    cafe_path = DATA_DIR / "noise" / "cafeteria.wav"

    s, sr = load16k(str(clean_path))
    v_fan, _ = load16k(str(fan_path))
    v_cafe, _ = load16k(str(cafe_path))

    metrics_rows = []

    # -------------------------------------------------------------
    # TN1: Input SNR (0, 5, 10 dB)
    # -------------------------------------------------------------
    print("\n--- TN1: Khảo sát Mức Nhiễu Đầu Vào (Input SNR: 0, 5, 10 dB) ---")
    tn1_results = []
    for snr_target in [0.0, 5.0, 10.0]:
        y_noisy, _ = add_noise_at_snr(s, v_fan, snr_target)
        x_ss, x_w, stats = enhance_both(y_noisy, v_fan, clean_ref=s, sr=sr)
        tn1_results.append({
            "TN": "TN1",
            "Param": f"Input SNR = {snr_target}dB",
            "SNR_in": stats["snr_in"],
            "SNR_out_SS": stats["snr_out_ss"],
            "Delta_SS": stats["delta_ss"],
            "SNR_out_Wiener": stats["snr_out_wiener"],
            "Delta_Wiener": stats["delta_wiener"]
        })
        print(f"  Target: {snr_target:2.0f}dB -> Meas: {stats['snr_in']:5.2f}dB | SS: {stats['snr_out_ss']:5.2f}dB (Delta: +{stats['delta_ss']:4.2f}dB) | Wiener: {stats['snr_out_wiener']:5.2f}dB (Delta: +{stats['delta_wiener']:4.2f}dB)")
    metrics_rows.extend(tn1_results)

    # -------------------------------------------------------------
    # TN2: Spectral Subtraction Over-subtraction beta (1.0, 1.5, 2.0)
    # -------------------------------------------------------------
    print("\n--- TN2: Khảo sát Hệ số Over-subtraction beta (1.0, 1.5, 2.0 tại SNR=5dB) ---")
    y_5db, _ = add_noise_at_snr(s, v_fan, 5.0)
    tn2_results = []
    for beta_val in [1.0, 1.5, 2.0]:
        x_ss = enhance_spectral_subtraction(y_5db, v_fan, sr=sr, beta=beta_val, floor=0.02)
        snr_out = snr_db(s, x_ss)
        delta_snr = snr_out - snr_db(s, y_5db)
        tn2_results.append({
            "TN": "TN2",
            "Param": f"beta = {beta_val}",
            "SNR_in": 5.0,
            "SNR_out_SS": snr_out,
            "Delta_SS": delta_snr,
            "SNR_out_Wiener": None,
            "Delta_Wiener": None
        })
        print(f"  beta = {beta_val:.1f} -> SNR_out = {snr_out:.2f} dB | Delta_SNR = +{delta_snr:.2f} dB (Musical noise tăng khi beta tăng)")
    metrics_rows.extend(tn2_results)

    # -------------------------------------------------------------
    # TN3: Spectral Floor gamma (0.0, 0.01, 0.05)
    # -------------------------------------------------------------
    print("\n--- TN3: Khảo sát Sàn Phổ Spectral Floor gamma (0.0, 0.01, 0.05 tại beta=1.5) ---")
    tn3_results = []
    for floor_val in [0.0, 0.01, 0.05]:
        x_ss = enhance_spectral_subtraction(y_5db, v_fan, sr=sr, beta=1.5, floor=floor_val)
        snr_out = snr_db(s, x_ss)
        delta_snr = snr_out - snr_db(s, y_5db)
        tn3_results.append({
            "TN": "TN3",
            "Param": f"floor gamma = {floor_val}",
            "SNR_in": 5.0,
            "SNR_out_SS": snr_out,
            "Delta_SS": delta_snr,
            "SNR_out_Wiener": None,
            "Delta_Wiener": None
        })
        print(f"  gamma = {floor_val:.2f} -> SNR_out = {snr_out:.2f} dB | Delta_SNR = +{delta_snr:.2f} dB (Floor > 0 hạn chế musical noise)")
    metrics_rows.extend(tn3_results)

    # -------------------------------------------------------------
    # TN4: Feature Speaker: MFCC (40-D) vs MFCC + Delta (80-D)
    # -------------------------------------------------------------
    print("\n--- TN4: Khảo sát Bộ Đặc Trưng Speaker-ID (MFCC-only vs MFCC+Delta) ---")
    speakers = ["spk01", "spk02", "spk03"]
    for feat_name, inc_delta in [("MFCC_only (40D)", False), ("MFCC_plus_Delta (80D)", True)]:
        enr_dict = {spk: [] for spk in speakers}
        for spk in speakers:
            for i in range(1, 5):
                audio, _ = load16k(str(DATA_DIR / "clean" / f"{spk}_enroll_{i:02d}.wav"))
                emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=inc_delta, use_cmn=False)
                enr_dict[spk].append(emb)

        test_dict = {spk: [] for spk in speakers}
        for spk in speakers:
            for i in range(1, 3):
                audio, _ = load16k(str(DATA_DIR / "clean" / f"{spk}_test_{i:02d}.wav"))
                emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=inc_delta, use_cmn=False)
                test_dict[spk].append(emb)

        models = train_speaker_models(enr_dict)
        res = evaluate_identification(test_dict, models)
        metrics_rows.append({
            "TN": "TN4",
            "Param": feat_name,
            "Accuracy": res["accuracy"],
            "Avg_Margin": res["avg_margin"]
        })
        print(f"  {feat_name:22s} -> Accuracy = {res['accuracy']*100:.1f}% | Avg Margin = {res['avg_margin']:.4f}")

    # -------------------------------------------------------------
    # TN5: Cepstral Mean Normalization (CMN: Off vs On)
    # -------------------------------------------------------------
    print("\n--- TN5: Khảo sát Chuẩn Hóa Phổ CMN (Off vs On) ---")
    for cmn_name, cmn_flag in [("CMN_Off", False), ("CMN_On", True)]:
        enr_dict = {spk: [] for spk in speakers}
        for spk in speakers:
            for i in range(1, 5):
                audio, _ = load16k(str(DATA_DIR / "clean" / f"{spk}_enroll_{i:02d}.wav"))
                emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=cmn_flag)
                enr_dict[spk].append(emb)

        test_dict = {spk: [] for spk in speakers}
        for spk in speakers:
            for i in range(1, 3):
                audio, _ = load16k(str(DATA_DIR / "clean" / f"{spk}_test_{i:02d}.wav"))
                emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=cmn_flag)
                test_dict[spk].append(emb)

        models = train_speaker_models(enr_dict)
        res = evaluate_identification(test_dict, models)
        metrics_rows.append({
            "TN": "TN5",
            "Param": cmn_name,
            "Accuracy": res["accuracy"],
            "Avg_Margin": res["avg_margin"]
        })
        print(f"  {cmn_name:10s} -> Accuracy = {res['accuracy']*100:.1f}% | Avg Margin = {res['avg_margin']:.4f}")

    # -------------------------------------------------------------
    # TN6: Số Lượng Mẫu Enrollment (1, 2, 4 câu thoại)
    # -------------------------------------------------------------
    print("\n--- TN6: Khảo sát Số Lượng Mẫu Enrollment (Tokens = 1, 2, 4) ---")
    for n_tok in [1, 2, 4]:
        enr_dict = {spk: [] for spk in speakers}
        for spk in speakers:
            for i in range(1, n_tok + 1):
                audio, _ = load16k(str(DATA_DIR / "clean" / f"{spk}_enroll_{i:02d}.wav"))
                emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=False)
                enr_dict[spk].append(emb)

        test_dict = {spk: [] for spk in speakers}
        for spk in speakers:
            for i in range(1, 3):
                audio, _ = load16k(str(DATA_DIR / "clean" / f"{spk}_test_{i:02d}.wav"))
                emb = extract_speaker_embedding(audio, sr=sr, n_mfcc=20, include_delta=True, use_cmn=False)
                test_dict[spk].append(emb)

        models = train_speaker_models(enr_dict)
        res = evaluate_identification(test_dict, models)
        metrics_rows.append({
            "TN": "TN6",
            "Param": f"Tokens = {n_tok}",
            "Accuracy": res["accuracy"],
            "Avg_Margin": res["avg_margin"]
        })
        print(f"  Enroll Tokens = {n_tok} -> Accuracy = {res['accuracy']*100:.1f}% | Avg Margin = {res['avg_margin']:.4f}")

    # -------------------------------------------------------------
    # TN7: Ma Trận Trộn A (Dễ: det=0.76 vs Gần suy biến: det~0.05)
    # -------------------------------------------------------------
    print("\n--- TN7: Khảo sát Ma Trận Trộn A (Dễ vs Gần Suy Biến) ---")
    srcA_path = DATA_DIR / "separation" / "source_A.wav"
    srcB_path = DATA_DIR / "separation" / "source_B.wav"
    s1, _ = load16k(str(srcA_path))
    s2, _ = load16k(str(srcB_path))

    A_easy = np.array([[1.0, 0.6], [0.4, 1.0]])
    A_ill = np.array([[1.0, 0.95], [0.90, 0.90]])  # det = 1.0*0.90 - 0.95*0.90 = 0.045

    res_easy = separate_sources_pipeline(s1, s2, A=A_easy)
    res_ill = separate_sources_pipeline(s1, s2, A=A_ill)

    metrics_rows.append({
        "TN": "TN7",
        "Param": "A_standard (det=0.76)",
        "det_A": res_easy["det_A"],
        "Corr_SrcA": res_easy["aligned_correlations"][0],
        "Corr_SrcB": res_easy["aligned_correlations"][1],
        "Avg_Corr": np.mean(res_easy["aligned_correlations"])
    })
    metrics_rows.append({
        "TN": "TN7",
        "Param": "A_ill_conditioned (det~0.045)",
        "det_A": res_ill["det_A"],
        "Corr_SrcA": res_ill["aligned_correlations"][0],
        "Corr_SrcB": res_ill["aligned_correlations"][1],
        "Avg_Corr": np.mean(res_ill["aligned_correlations"])
    })
    print(f"  A dễ (det={res_easy['det_A']:.4f}) -> Corr A: {res_easy['aligned_correlations'][0]:.4f}, Corr B: {res_easy['aligned_correlations'][1]:.4f}")
    print(f"  A khó (det={res_ill['det_A']:.4f})  -> Corr A: {res_ill['aligned_correlations'][0]:.4f}, Corr B: {res_ill['aligned_correlations'][1]:.4f} (Separation suy giảm nặng)")

    # -------------------------------------------------------------
    # TN8: ICA Trong Môi Trường Nhiễu (Clean Mix vs Noisy Mix)
    # -------------------------------------------------------------
    print("\n--- TN8: Khảo sát ICA Trong Môi Trường Nhiễu (Clean Mix vs Noisy Mix) ---")
    for noise_std, name in [(0.0, "Clean Mix (std=0)"), (0.05, "Light Noise (std=0.05)"), (0.15, "Heavy Noise (std=0.15)")]:
        res_noise = separate_sources_pipeline(s1, s2, A=A_easy, add_noise_std=noise_std)
        metrics_rows.append({
            "TN": "TN8",
            "Param": name,
            "Noise_Std": noise_std,
            "Corr_SrcA": res_noise["aligned_correlations"][0],
            "Corr_SrcB": res_noise["aligned_correlations"][1],
            "Avg_Corr": np.mean(res_noise["aligned_correlations"])
        })
        print(f"  {name:25s} -> Corr A: {res_noise['aligned_correlations'][0]:.4f}, Corr B: {res_noise['aligned_correlations'][1]:.4f}")

    # Xuất toàn bộ kết quả thí nghiệm ra CSV
    df = pd.DataFrame(metrics_rows)
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[+] Đã xuất toàn bộ kết quả đo đạc ra file: {csv_path.relative_to(BASE_DIR)}")


def main():
    print("=" * 80)
    print("KHỞI CHẠY HỆ THỐNG THỰC NGHIỆM TỰ ĐỘNG - LAB 5 (CSE457)")
    print("=" * 80)

    # 1. Chạy Case study
    case_res = run_case_study()

    # 2. Chạy 8 Thí nghiệm bắt buộc
    run_mandatory_experiments()

    print("\n" + "=" * 80)
    print("HOÀN THÀNH TOÀN BỘ CÁC BƯỚC THỰC NGHIỆM VÀ XUẤT KẾT QUẢ ĐỊNH LƯỢNG!")
    print(f"Tất cả file âm thanh: {AUDIO_OUT_DIR.relative_to(BASE_DIR)}")
    print(f"Tất cả biểu đồ hình ảnh: {FIGS_OUT_DIR.relative_to(BASE_DIR)}")
    print(f"Bảng tổng hợp số liệu: {(RESULTS_DIR / 'metrics_summary.csv').relative_to(BASE_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
