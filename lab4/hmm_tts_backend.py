"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4
Mô-đun HMM-based Parametric Synthesis (Mô phỏng Sinh Tham số Âm học)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Nguyên lý HMM Generation:
  1. Text -> Chuỗi Token -> Chuỗi Trạng thái HMM (State Sequence Q)
  2. Mỗi state mang phân bố xác suất: F0, Duration, Formant/Spectral parameters.
  3. Áp dụng ràng buộc gia tốc động (Delta features) để làm trơn quỹ đạo (Parameter Trajectory)
     thay vì các bậc thang rời rạc thô.
  4. Vocoder tham số (Source-Filter) tái tạo waveform từ quỹ đạo F0 và phổ.
===============================================================================
"""

import numpy as np
import scipy.signal as signal
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

# Bản đồ ngữ điệu và formant trung bình cho các từ vựng xếp hàng
PHONETIC_MODELS = {
    "mời":   {"base_f0": 145, "delta_f0": -15, "states": 4, "f1": 350,  "f2": 1100, "dur_frames": 24},
    "khách": {"base_f0": 175, "delta_f0": +20, "states": 4, "f1": 650,  "f2": 1800, "dur_frames": 20},
    "hàng":  {"base_f0": 140, "delta_f0": -20, "states": 4, "f1": 700,  "f2": 1300, "dur_frames": 22},
    "số":    {"base_f0": 180, "delta_f0": +25, "states": 3, "f1": 450,  "f2": 1000, "dur_frames": 18},
    "đến":   {"base_f0": 170, "delta_f0": +15, "states": 3, "f1": 500,  "f2": 1900, "dur_frames": 18},
    "quầy":  {"base_f0": 142, "delta_f0": -18, "states": 4, "f1": 400,  "f2": 1200, "dur_frames": 22},
    "không": {"base_f0": 155, "delta_f0": 0,   "states": 4, "f1": 450,  "f2": 950,  "dur_frames": 20},
    "một":   {"base_f0": 135, "delta_f0": -25, "states": 3, "f1": 480,  "f2": 1150, "dur_frames": 16},
    "hai":   {"base_f0": 160, "delta_f0": 0,   "states": 3, "f1": 750,  "f2": 1500, "dur_frames": 20},
    "ba":    {"base_f0": 158, "delta_f0": 0,   "states": 3, "f1": 800,  "f2": 1350, "dur_frames": 20},
    "bốn":   {"base_f0": 182, "delta_f0": +22, "states": 3, "f1": 500,  "f2": 1050, "dur_frames": 18},
    "năm":   {"base_f0": 160, "delta_f0": 0,   "states": 3, "f1": 720,  "f2": 1400, "dur_frames": 20},
    "sáu":   {"base_f0": 185, "delta_f0": +25, "states": 3, "f1": 520,  "f2": 1100, "dur_frames": 18},
    "bảy":   {"base_f0": 148, "delta_f0": -12, "states": 3, "f1": 680,  "f2": 1650, "dur_frames": 18},
    "tám":   {"base_f0": 184, "delta_f0": +24, "states": 3, "f1": 760,  "f2": 1400, "dur_frames": 18},
    "chín":  {"base_f0": 188, "delta_f0": +26, "states": 3, "f1": 380,  "f2": 2100, "dur_frames": 18},
}

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}


def normalize_text_to_tokens(text):
    clean = text.lower().replace(".", "").replace(",", "").replace("?", "")
    tokens = []
    for tok in clean.split():
        if tok.isdigit():
            tokens.extend(DIGIT_MAP[d] for d in tok if d in DIGIT_MAP)
        else:
            tokens.append(tok)
    return tokens


def generate_hmm_trajectories(tokens, duration_scale=1.0, f0_scale=1.0, frame_shift_ms=10):
    """
    Sinh quỹ đạo tham số âm học (Trajectory generation) từ mô hình HMM.
    Mô phỏng cả 2 trường hợp:
      - Raw Step Trajectory (không có delta feature - gián đoạn bậc thang)
      - Smooth Trajectory (với delta feature MLPG - đường cong mượt mà)
    """
    f0_steps = []
    f1_steps = []
    f2_steps = []
    state_sequence = []
    current_state_id = 0

    for tok in tokens:
        model = PHONETIC_MODELS.get(tok, {"base_f0": 150, "delta_f0": 0, "states": 3, "f1": 500, "f2": 1500, "dur_frames": 18})
        n_states = model["states"]
        total_frames = int(model["dur_frames"] * duration_scale)
        frames_per_state = max(2, total_frames // n_states)

        base_f0 = model["base_f0"] * f0_scale
        delta_f0 = model["delta_f0"] * f0_scale
        f1_base = model["f1"]
        f2_base = model["f2"]

        for s in range(n_states):
            current_state_id += 1
            progress = s / max(1, n_states - 1)
            # F0 và formant trung bình trong state
            state_f0 = base_f0 + delta_f0 * (progress - 0.5)
            state_f1 = f1_base * (1.0 + 0.15 * np.sin(np.pi * progress))
            state_f2 = f2_base * (1.0 - 0.10 * np.cos(np.pi * progress))

            for _ in range(frames_per_state):
                f0_steps.append(state_f0)
                f1_steps.append(state_f1)
                f2_steps.append(state_f2)
                state_sequence.append(current_state_id)

    f0_steps = np.array(f0_steps, dtype=np.float32)
    f1_steps = np.array(f1_steps, dtype=np.float32)
    f2_steps = np.array(f2_steps, dtype=np.float32)

    # Làm trơn bằng bộ lọc đạo hàm bậc 1 mô phỏng giải thuật MLPG khi có Delta-feature
    f0_smooth = gaussian_filter1d(f0_steps, sigma=2.5)
    f1_smooth = gaussian_filter1d(f1_steps, sigma=2.5)
    f2_smooth = gaussian_filter1d(f2_steps, sigma=2.5)

    time_axis = np.arange(len(f0_smooth)) * (frame_shift_ms / 1000.0)

    tracks = {
        "time": time_axis,
        "state_sequence": state_sequence,
        "f0_step": f0_steps,
        "f0": f0_smooth,
        "f1_step": f1_steps,
        "f1": f1_smooth,
        "f2": f2_smooth,
        "tokens": tokens,
    }
    return tracks


def source_filter_vocoder(tracks, sr=16000, frame_shift_ms=10):
    """
    Source-Filter Resynthesis:
      - Nguồn kích thích (Source): Chuỗi xung tuần hoàn theo chu kỳ pitch F0(t).
      - Bộ lọc (Filter): 2 bộ cộng hưởng formant (F1, F2).
    """
    total_frames = len(tracks["f0"])
    hop_length = int(sr * (frame_shift_ms / 1000.0))
    total_samples = total_frames * hop_length

    # 1. Sinh pha tích lũy để tạo nguồn kích thích f0 biến thiên
    f0_interpolated = np.interp(
        np.linspace(0, total_frames - 1, total_samples),
        np.arange(total_frames),
        tracks["f0"]
    )
    phase = 2.0 * np.pi * np.cumsum(f0_interpolated) / sr
    excitation = signal.sawtooth(phase)  # Sóng răng cưa giàu hài âm mô phỏng dây thanh

    # 2. Lọc Formant động qua Resonator
    f1_interp = np.interp(
        np.linspace(0, total_frames - 1, total_samples),
        np.arange(total_frames),
        tracks["f1"]
    )
    f2_interp = np.interp(
        np.linspace(0, total_frames - 1, total_samples),
        np.arange(total_frames),
        tracks["f2"]
    )

    mean_f1 = np.mean(f1_interp)
    mean_f2 = np.mean(f2_interp)

    # Bộ lọc cộng hưởng 2 cực cho F1 & F2
    bw = 80.0  # Băng thông formant ~80Hz
    b_f1, a_f1 = signal.iirpeak(mean_f1, Q=mean_f1 / bw, fs=sr)
    b_f2, a_f2 = signal.iirpeak(mean_f2, Q=mean_f2 / (bw * 1.5), fs=sr)

    filtered_f1 = signal.lfilter(b_f1, a_f1, excitation)
    filtered_f2 = signal.lfilter(b_f2, a_f2, excitation)

    waveform = 0.6 * filtered_f1 + 0.4 * filtered_f2
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8) * 0.8
    return waveform.astype(np.float32)


def synthesize_hmm(text, duration_scale=1.0, f0_scale=1.0, sr=16000):
    """
    API chính theo yêu cầu đề bài (Mục 5.5):
    y_hmm, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
    """
    tokens = normalize_text_to_tokens(text)
    tracks = generate_hmm_trajectories(tokens, duration_scale=duration_scale, f0_scale=f0_scale)
    waveform = source_filter_vocoder(tracks, sr=sr)
    return waveform, sr, tracks


if __name__ == "__main__":
    test_text = "Mời khách hàng số 23 đến quầy 5"
    y_hmm, sr, tracks = synthesize_hmm(test_text)
    sf.write("hmm_test.wav", y_hmm, sr)
    print(f"[+] Đã tổng hợp HMM thành công: {len(y_hmm)/sr:.2f}s, {len(tracks['f0'])} frames.")
