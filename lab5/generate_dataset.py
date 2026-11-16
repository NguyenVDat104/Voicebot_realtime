"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
Mô-đun Tổng hợp Bộ Dữ liệu Chuẩn Đa Người Nói & Tiếng Ồn (Dataset Generator)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Cấu trúc chuẩn sinh ra:
lab5_data/
├── clean/
│   ├── spk01_enroll_01.wav ... spk01_enroll_04.wav, spk01_test_01.wav, spk01_test_02.wav
│   ├── spk02_enroll_01.wav ... spk02_enroll_04.wav, spk02_test_01.wav, spk02_test_02.wav
│   └── spk03_enroll_01.wav ... spk03_enroll_04.wav, spk03_test_01.wav, spk03_test_02.wav
├── noise/
│   ├── fan.wav
│   └── cafeteria.wav
└── separation/
    ├── source_A.wav
    └── source_B.wav

Quy chuẩn:
- 16,000 Hz, Mono, PCM 16-bit WAV.
- Thời lượng câu thoại: 2.5 - 4.0 giây.
===============================================================================
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from gtts import gTTS

TARGET_SR = 16000

# 3 Người nói với câu thoại và đặc tính thanh điệu/cao độ/tốc độ khác biệt rõ nét
SPEAKER_CONFIGS = {
    "spk01": {
        "name": "Speaker 1 (Nam trầm)",
        "pitch_steps": -3.5,
        "stretch": 0.98,
        "enroll": [
            "Xin chào các bạn, hôm nay chúng ta sẽ bắt đầu buổi họp kỹ thuật xử lý tín hiệu.",
            "Tôi đề nghị xem xét lại tiến độ của hệ thống nhận dạng âm thanh trong tuần này.",
            "Phương pháp trừ phổ giúp cải thiện đáng kể tỷ số tín hiệu trên nhiễu âm thanh.",
            "Mô hình thống kê người nói được xây dựng dựa trên trung bình các vector đặc trưng.",
        ],
        "test": [
            "Thời tiết hôm nay tại Hà Nội rất mát mẻ và thuận lợi cho công việc nghiên cứu.",
            "Hệ thống trợ lý phòng họp thông minh đang hoạt động ổn định và chính xác.",
        ]
    },
    "spk02": {
        "name": "Speaker 2 (Trung tính)",
        "pitch_steps": 0.0,
        "stretch": 1.02,
        "enroll": [
            "Kính chào quý thầy cô và các bạn sinh viên đã tham dự buổi thực hành chuyên sâu.",
            "Bản phân tích chất lượng âm thanh sau khi lọc nhiễu cho kết quả rất khả quan.",
            "Thuật toán lọc Wiener hạn chế tối đa hiện tượng méo tiếng và âm thanh nhân tạo.",
            "Đề nghị mọi người kiểm tra lại các tham số cấu hình trước khi lưu kết quả.",
        ],
        "test": [
            "Xin cảm ơn các bạn đã chú ý lắng nghe bài thuyết trình của nhóm chúng tôi.",
            "Chúng tôi đã hoàn thành việc tách nguồn tín hiệu bằng giải thuật phân tích độc lập.",
        ]
    },
    "spk03": {
        "name": "Speaker 3 (Nữ cao)",
        "pitch_steps": +3.8,
        "stretch": 1.06,
        "enroll": [
            "Chào mọi người, tôi xin phép trình bày về mô hình nhận dạng tiếng nói tự động.",
            "Hiện tượng tiếng ồn âm nhạc xuất hiện khi hệ số trừ phổ được thiết lập quá lớn.",
            "Ma trận tương quan cho phép giải quyết triệt để sự bất định hoán vị nguồn âm.",
            "Chúng ta cần đánh giá độ chính xác của hệ thống trên tập dữ liệu kiểm thử độc lập.",
        ],
        "test": [
            "Kết quả đo đạc thực nghiệm cho thấy tỷ số tín hiệu trên nhiễu được cải thiện rõ.",
            "Dự án phát triển phần mềm trí tuệ nhân tạo đang đạt đúng chỉ tiêu đề ra.",
        ]
    }
}


def synthesize_clean_utterance(text, pitch_steps=0.0, stretch=1.0, target_sr=TARGET_SR):
    """Tổng hợp câu thoại tiếng Việt qua gTTS và biến đổi cao độ/tốc độ."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        tts = gTTS(text=text, lang="vi", slow=False)
        tts.save(tmp_path)
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)

        # Biến đổi tốc độ
        if abs(stretch - 1.0) > 0.01:
            y_proc = librosa.effects.time_stretch(y_trimmed, rate=stretch)
        else:
            y_proc = y_trimmed

        # Biến đổi cao độ
        if abs(pitch_steps) > 0.1:
            y_proc = librosa.effects.pitch_shift(y_proc, sr=target_sr, n_steps=pitch_steps)

        # Chuẩn hóa biên độ tối đa
        peak = np.max(np.abs(y_proc))
        if peak > 0:
            y_proc = (y_proc / peak) * 0.90

        # Thêm 0.15s khoảng lặng nhẹ hai đầu
        lead_trail = int(0.15 * target_sr)
        silence = np.random.normal(0, 1e-4, lead_trail).astype(np.float32)
        y_final = np.concatenate([silence, y_proc.astype(np.float32), silence])
        return y_final
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_fan_noise(duration_sec=7.0, sr=TARGET_SR):
    """
    Tạo tiếng ồn quạt làm mát (fan noise):
    - Động cơ quay: hài 50Hz, 100Hz, 150Hz.
    - Tiếng gió rít: nhiễu hồng (pink noise) qua bộ lọc thông thấp ~700 Hz.
    - Điều chế biên độ chu kỳ cánh quạt (~2 Hz).
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr
    rng = np.random.default_rng(101)

    # 1. Thành phần hài động cơ (motor hum)
    hum = (0.25 * np.sin(2 * np.pi * 50 * t) +
           0.15 * np.sin(2 * np.pi * 100 * t) +
           0.08 * np.sin(2 * np.pi * 150 * t))

    # 2. Thành phần luồng khí (pink/brown airflow noise)
    white = rng.normal(0, 1.0, n_samples)
    b, a = signal.butter(4, 700 / (sr / 2), btype='low')
    wind = signal.lfilter(b, a, white)
    wind = wind / (np.std(wind) + 1e-12)

    # 3. Điều chế biên độ cánh quạt nhẹ (~2.5 Hz)
    modulation = 0.85 + 0.15 * np.sin(2 * np.pi * 2.5 * t)

    noise = (hum + 0.9 * wind) * modulation
    noise = noise / np.max(np.abs(noise)) * 0.75
    return noise.astype(np.float32)


def generate_cafeteria_noise(duration_sec=7.0, sr=TARGET_SR):
    """
    Tạo tiếng ồn quán café/phòng ăn đông người (cafeteria babble noise):
    - Tạp âm lời nói nền (babble speech) trong dải tần 300 - 3400 Hz.
    - Tiếng lách cách bát đĩa/va chạm ngẫu nhiên (clatter transients).
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr
    rng = np.random.default_rng(202)

    # 1. Rì rầm đám đông (crowd babble approximation)
    white = rng.normal(0, 1.0, n_samples)
    b, a = signal.butter(4, [250 / (sr / 2), 3200 / (sr / 2)], btype='bandpass')
    babble = signal.lfilter(b, a, white)
    babble = babble / (np.std(babble) + 1e-12)

    # Thêm dao động nhiều nguồn nói
    flutter = (0.7 + 0.15 * np.sin(2 * np.pi * 1.8 * t) +
               0.15 * np.sin(2 * np.pi * 3.4 * t + 0.5))
    babble = babble * flutter

    # 2. Tiếng chạm đĩa chén (clatter transients)
    clatter = np.zeros(n_samples, dtype=np.float32)
    n_clatters = int(duration_sec * 3)  # ~3 tiếng va chạm mỗi giây
    click_indices = rng.choice(n_samples - 800, size=n_clatters, replace=False)
    for idx in click_indices:
        decay = np.exp(-np.linspace(0, 5, 600))
        freq = rng.uniform(2500, 5000)
        click = np.sin(2 * np.pi * freq * np.arange(600) / sr) * decay
        clatter[idx:idx+600] += rng.uniform(0.3, 0.7) * click

    noise = 0.8 * babble + 0.4 * clatter
    noise = noise / np.max(np.abs(noise)) * 0.75
    return noise.astype(np.float32)


def generate_all():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "lab5_data"
    clean_dir = data_dir / "clean"
    noise_dir = data_dir / "noise"
    sep_dir = data_dir / "separation"

    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    sep_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BẮT ĐẦU TỔNG HỢP BỘ DỮ LIỆU LAB 5: MULTI-SPEAKER, NOISE & SEPARATION")
    print(f"Thư mục lưu trữ: {data_dir}")
    print("=" * 70)

    # 1. Sinh các file Clean Speech cho 3 Speakers
    generated_files = []
    for spk_id, cfg in SPEAKER_CONFIGS.items():
        print(f"\n[+] Đang tạo âm thanh cho {spk_id} ({cfg['name']})...")
        
        # 4 Enroll files
        for i, text in enumerate(cfg["enroll"], start=1):
            filename = f"{spk_id}_enroll_{i:02d}.wav"
            out_path = clean_dir / filename
            audio = synthesize_clean_utterance(text, pitch_steps=cfg["pitch_steps"], stretch=cfg["stretch"])
            sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
            dur = len(audio) / TARGET_SR
            generated_files.append((str(out_path.relative_to(base_dir)), dur, "clean_enroll"))
            print(f"    - {filename} ({dur:.2f}s): \"{text[:40]}...\"")

        # 2 Test files
        for i, text in enumerate(cfg["test"], start=1):
            filename = f"{spk_id}_test_{i:02d}.wav"
            out_path = clean_dir / filename
            audio = synthesize_clean_utterance(text, pitch_steps=cfg["pitch_steps"], stretch=cfg["stretch"])
            sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
            dur = len(audio) / TARGET_SR
            generated_files.append((str(out_path.relative_to(base_dir)), dur, "clean_test"))
            print(f"    - {filename} ({dur:.2f}s): \"{text[:40]}...\"")

    # 2. Sinh các file Noise
    print("\n[+] Đang tổng hợp các tệp âm thanh tiếng ồn (Noise profiles)...")
    fan_audio = generate_fan_noise(duration_sec=7.0, sr=TARGET_SR)
    fan_path = noise_dir / "fan.wav"
    sf.write(fan_path, fan_audio, TARGET_SR, subtype="PCM_16")
    dur_fan = len(fan_audio) / TARGET_SR
    generated_files.append((str(fan_path.relative_to(base_dir)), dur_fan, "noise_fan"))
    print(f"    - fan.wav ({dur_fan:.2f}s): Tiếng quạt gió làm mát (50Hz hum + pink noise).")

    cafe_audio = generate_cafeteria_noise(duration_sec=7.0, sr=TARGET_SR)
    cafe_path = noise_dir / "cafeteria.wav"
    sf.write(cafe_path, cafe_audio, TARGET_SR, subtype="PCM_16")
    dur_cafe = len(cafe_audio) / TARGET_SR
    generated_files.append((str(cafe_path.relative_to(base_dir)), dur_cafe, "noise_cafeteria"))
    print(f"    - cafeteria.wav ({dur_cafe:.2f}s): Tiếng ồn quán ăn/phòng hội nghị (crowd babble + clatter).")

    # 3. Sinh 2 nguồn cho Separation (cắt/pad chính xác cùng độ dài 3.5 giây = 56,000 mẫu)
    print("\n[+] Đang tạo 2 nguồn tín hiệu cho bài toán phân tách nguồn (Separation)...")
    target_sep_len = int(3.5 * TARGET_SR)  # 56,000 mẫu

    # Nguồn A: spk01 nói một câu kỹ thuật
    text_A = "Chúng ta cần bảo đảm hệ thống xử lý tín hiệu đạt tiêu chuẩn độ trễ thấp."
    audio_A = synthesize_clean_utterance(text_A, pitch_steps=SPEAKER_CONFIGS["spk01"]["pitch_steps"], stretch=0.98)
    if len(audio_A) > target_sep_len:
        audio_A = audio_A[:target_sep_len]
    else:
        audio_A = np.pad(audio_A, (0, target_sep_len - len(audio_A)))

    # Nguồn B: spk03 nói một câu phân tích
    text_B = "Giải thuật tách nguồn mù phân tách độc lập các tín hiệu âm thanh chồng lấn."
    audio_B = synthesize_clean_utterance(text_B, pitch_steps=SPEAKER_CONFIGS["spk03"]["pitch_steps"], stretch=1.05)
    if len(audio_B) > target_sep_len:
        audio_B = audio_B[:target_sep_len]
    else:
        audio_B = np.pad(audio_B, (0, target_sep_len - len(audio_B)))

    # Chuẩn hóa RMS về 0.08 theo chuẩn đề bài
    def normalize_rms(x, target=0.08):
        cur_rms = np.sqrt(np.mean(x**2)) + 1e-12
        return x * (target / cur_rms)

    audio_A = normalize_rms(audio_A, 0.08)
    audio_B = normalize_rms(audio_B, 0.08)

    path_A = sep_dir / "source_A.wav"
    path_B = sep_dir / "source_B.wav"
    sf.write(path_A, audio_A.astype(np.float32), TARGET_SR, subtype="PCM_16")
    sf.write(path_B, audio_B.astype(np.float32), TARGET_SR, subtype="PCM_16")

    generated_files.append((str(path_A.relative_to(base_dir)), len(audio_A) / TARGET_SR, "sep_source_A"))
    generated_files.append((str(path_B.relative_to(base_dir)), len(audio_B) / TARGET_SR, "sep_source_B"))
    print(f"    - source_A.wav ({len(audio_A)/TARGET_SR:.2f}s, RMS={np.sqrt(np.mean(audio_A**2)):.4f})")
    print(f"    - source_B.wav ({len(audio_B)/TARGET_SR:.2f}s, RMS={np.sqrt(np.mean(audio_B**2)):.4f})")

    print("\n" + "=" * 70)
    print(f"HOÀN TẤT TỔNG HỢP: Tổng cộng {len(generated_files)} file WAV đã được chuẩn hóa.")
    print("=" * 70)


if __name__ == "__main__":
    generate_all()
