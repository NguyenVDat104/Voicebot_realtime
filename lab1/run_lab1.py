"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
LAB 1: PHÂN TÍCH VÀ XỬ LÝ TÍN HIỆU ÂM THANH SỐ
Pipeline thực nghiệm tự động: Từ tín hiệu rời rạc, FFT/STFT, Lọc số, Lượng tử hóa đến Mã hóa

Tệp: run_lab1.py
Mục đích:
  - Tự động thực thi toàn bộ 7 khối bài tập (A -> G) theo đúng yêu cầu đề cương.
  - Sinh toàn bộ tệp âm thanh đầu ra lưu vào thư mục `audio/`.
  - Xuất toàn bộ biểu đồ trực quan hóa lưu vào thư mục `figures/`.
  - In báo cáo thống kê các chỉ số kỹ thuật định lượng ra màn hình console.
  - Từng khối lệnh đều có chú thích (comments) cặn kẽ giải thích nguyên lý hoạt động,
    công thức toán học và lý do chọn lựa tham số kỹ thuật.
===============================================================================
"""

import os
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Thiết lập backend matplotlib không tương tác để vẽ đồ thị và lưu file ổn định
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Đường dẫn thư mục làm việc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("KHỞI CHẠY PIPELINE THỰC NGHIỆM LAB 1 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI")
print(f"Thư mục gốc Lab 1: {BASE_DIR}")
print("=" * 80)


# =============================================================================
# BƯỚC 0: TẠO/CHUẨN BỊ TỆP ÂM THANH ĐẦU VÀO ĐA DẢI TẦN (input.wav)
# =============================================================================
def generate_rich_input_audio(filepath, duration_sec=10.0, fs=44100):
    """
    Hàm sinh tín hiệu âm thanh đa dải tần nhân tạo (Stereo, 44.1 kHz, 16-bit PCM).
    Tín hiệu kết hợp:
      1. Thành phần nhạc điệu (Hòa âm đa tần số thấp & trung: 220Hz, 440Hz, 660Hz, 880Hz).
      2. Thành phần giả lập nguyên âm tiếng nói (Speech Formants F0=130Hz, F1=730Hz, F2=1090Hz, F3=2440Hz).
      3. Thành phần xung nhịp chuyển tiếp nhanh (Percussion Transient).
      4. Khoảng lặng âm (Silence / Pause) để phục vụ kiểm thử phân tích năng lượng đoạn tĩnh vs động.
      5. Hai kênh stereo khác nhau nhẹ về pha và biên độ để phân tích kênh trái/phải.
    """
    print(f"\n[BƯỚC 0] Đang tạo tệp âm thanh đầu vào mẫu chuẩn tại: {filepath}")
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    # 1. Đoạn 0s - 3s: Hòa âm nhạc cụ (A3 chord: 220Hz + harmonics)
    harmonics_left = (
        0.35 * np.sin(2 * np.pi * 220 * t) +
        0.25 * np.sin(2 * np.pi * 440 * t) +
        0.15 * np.sin(2 * np.pi * 660 * t) +
        0.10 * np.sin(2 * np.pi * 880 * t) +
        0.05 * np.sin(2 * np.pi * 1320 * t)
    )
    # Kênh phải có thêm âm sắc bổng 1760Hz và 3520Hz
    harmonics_right = (
        0.30 * np.sin(2 * np.pi * 220 * t) +
        0.25 * np.sin(2 * np.pi * 440 * t) +
        0.15 * np.sin(2 * np.pi * 880 * t) +
        0.12 * np.sin(2 * np.pi * 1760 * t) +
        0.08 * np.sin(2 * np.pi * 3520 * t)
    )
    
    # 2. Đoạn 3s - 6s: Giả lập nguyên âm tiếng nói /a/ với F0 và các Formant
    # F0 = 130 Hz (tần số cơ bản giọng nam), Formants: F1=730Hz, F2=1090Hz, F3=2440Hz, F4=3500Hz
    vowel_mod = np.where((t >= 3.0) & (t < 6.0), 1.0, 0.0)
    pulse_train = signal.sawtooth(2 * np.pi * 130 * t)
    vowel_sound = (
        0.40 * np.sin(2 * np.pi * 730 * t) * np.exp(-((t % 0.05)/0.02)) +
        0.25 * np.sin(2 * np.pi * 1090 * t) * np.exp(-((t % 0.05)/0.02)) +
        0.15 * np.sin(2 * np.pi * 2440 * t) +
        0.08 * np.sin(2 * np.pi * 3500 * t)
    ) * vowel_mod

    # 3. Đoạn 6s - 8s: Xung nhịp gõ nhịp điệu (Transient clicks tại mỗi 0.5s)
    beat_mod = np.where((t >= 6.0) & (t < 8.0), 1.0, 0.0)
    decay = np.exp(-((t % 0.5) / 0.03)) * beat_mod
    percussion = 0.5 * np.sin(2 * np.pi * 1200 * t) * decay + 0.3 * np.random.normal(0, 0.1, len(t)) * decay

    # 4. Đoạn 8s - 10s: Khoảng nghỉ / nhiễu nền mức rất thấp (Silence region)
    silence_mod = np.where(t >= 8.0, 0.02, 1.0)

    sig_left = (harmonics_left * (t < 3.0) + vowel_sound + percussion) * silence_mod
    sig_right = (harmonics_right * (t < 3.0) + vowel_sound + percussion) * silence_mod
    
    # Thêm một chút nhiễu trắng nhẹ để mô phỏng môi trường thực tế (-50 dB)
    sig_left += np.random.normal(0, 0.003, len(t))
    sig_right += np.random.normal(0, 0.003, len(t))
    
    # Chuẩn hóa biên độ cực đại đạt xấp xỉ 0.95 để tránh clipping
    max_val = max(np.max(np.abs(sig_left)), np.max(np.abs(sig_right)))
    sig_left = (sig_left / max_val) * 0.95
    sig_right = (sig_right / max_val) * 0.95
    
    # Chuyển đổi sang định dạng số nguyên 16-bit PCM ([-32768, 32767])
    stereo_16bit = np.column_stack((
        np.int16(sig_left * 32767),
        np.int16(sig_right * 32767)
    ))
    
    wavfile.write(filepath, fs, stereo_16bit)
    print(f"  -> Đã tạo thành công tệp đầu vào 16-bit stereo ({duration_sec}s, {fs}Hz).")


input_wav_path = os.path.join(AUDIO_DIR, "input.wav")
if not os.path.exists(input_wav_path):
    generate_rich_input_audio(input_wav_path)


# =============================================================================
# KHỐI A: ĐỌC VÀ KIỂM TRA DỮ LIỆU ÂM THANH (METADATA & PREPROCESSING)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI A] ĐỌC VÀ KIỂM TRA DỮ LIỆU ÂM THANH")
print("=" * 80)

"""
Giải thích nguyên lý Khối A:
1. Đọc tệp WAV bằng `scipy.io.wavfile.read`:
   - Trả về `Fs`: sampling rate (tần số lấy mẫu, đơn vị Hz), biểu diễn số mẫu thu được trong 1 giây.
   - Trả về mảng `data`: dạng mảng NumPy với shape=(N_samples, Channels).
2. Trích xuất Metadata:
   - Thời lượng (duration): Duration = Tổng số mẫu N / Fs (giây).
   - Bit depth: Dựa vào kiểu dữ liệu của mảng số nguyên (ví dụ int16 tương ứng 16 bit/mẫu).
3. Chuyển đổi sang Mono (Kênh đơn):
   - Khi tệp là stereo (2 kênh), tai người nhận tín hiệu trái (L) và phải (R).
   - Để phân tích toán học 1 chiều, chuẩn hóa kênh đơn: x_mono[n] = (x_L[n] + x_R[n]) / 2.
4. Chuẩn hóa miền giá trị về [-1.0, 1.0]:
   - Mẫu 16-bit nguyên nằm trong [-32768, 32767].
   - Chia cho 32768.0 (2^(B-1)) để đưa về số thực dấu phẩy động [-1.0, 1.0].
   - Giúp các phép biến đổi Fourier và lọc số không bị tràn số học và đồng nhất thang đo biên độ.
"""

Fs, raw_audio = wavfile.read(input_wav_path)
file_size_bytes = os.path.getsize(input_wav_path)
n_samples = raw_audio.shape[0]
n_channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[1]
duration = n_samples / Fs
bit_depth = 16 if raw_audio.dtype == np.int16 else (32 if raw_audio.dtype == np.int32 else 8)

print(f"  • Tệp âm thanh         : {os.path.basename(input_wav_path)}")
print(f"  • Tần số lấy mẫu (Fs)  : {Fs:,} Hz (Tần số Nyquist = {Fs/2:,} Hz)")
print(f"  • Số lượng kênh        : {n_channels} ({'Stereo' if n_channels == 2 else 'Mono'})")
print(f"  • Tổng số mẫu (N)      : {n_samples:,} samples")
print(f"  • Thời lượng (Duration): {duration:.3f} giây")
print(f"  • Kiểu dữ liệu (dtype) : {raw_audio.dtype} ({bit_depth} bit/mẫu)")
print(f"  • Kích thước tệp       : {file_size_bytes:,} bytes (~{file_size_bytes / (1024*1024):.2f} MB)")

# Chuyển đổi về kênh đơn (mono) và chuẩn hóa số thực về [-1.0, 1.0]
if n_channels == 2:
    left_channel = raw_audio[:, 0].astype(np.float64) / (2**(bit_depth - 1))
    right_channel = raw_audio[:, 1].astype(np.float64) / (2**(bit_depth - 1))
    x = (left_channel + right_channel) / 2.0
else:
    x = raw_audio.astype(np.float64) / (2**(bit_depth - 1))

print(f"  -> Hoàn thành chuẩn hóa biên độ mảng âm thanh về miền [-1.0, 1.0].")


# =============================================================================
# KHỐI B: PHÂN TÍCH MIỀN THỜI GIAN (TIME-DOMAIN ANALYSIS)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI B] PHÂN TÍCH MIỀN THỜI GIAN")
print("=" * 80)

"""
Giải thích nguyên lý Khối B:
1. Peak (Biên độ đỉnh): Peak = max |x[n]|.
   - Cho biết mức biên độ tức thời lớn nhất.
   - Kiểm tra hiện tượng Clipping (xén ngọn): nếu Peak >= 1.0 (hoặc sát 1.0), tín hiệu có nguy cơ
     bị méo phi tuyến do vượt ngưỡng dải động của bộ ADC/DAC.
2. RMS (Root Mean Square - Năng lượng hiệu dụng):
   - RMS = sqrt( (1/N) * sum(x[n]^2) ).
   - Đại diện cho cường độ cảm nhận âm lượng thực tế của tai người (perceived loudness).
   - Quy đổi sang thang dBFS (decibels relative to full scale, mốc 1.0 = 0 dBFS):
     RMS_dBFS = 20 * log10(RMS).
3. Tổng năng lượng (Energy): E = sum(x[n]^2).
   - Dùng để so sánh công suất giữa các phân đoạn âm thanh khác nhau.
"""

peak_val = np.max(np.abs(x))
rms_val = np.sqrt(np.mean(x**2))
rms_dbfs = 20 * np.log10(rms_val + 1e-12)
total_energy = np.sum(x**2)

print(f"  • Biên độ đỉnh (Peak)  : {peak_val:.4f} (Trạng thái: {'An toàn, không bị clipping' if peak_val < 0.99 else 'CẢNH BÁO CLIPPING!'})")
print(f"  • Năng lượng RMS       : {rms_val:.4f} ({rms_dbfs:.2f} dBFS)")
print(f"  • Tổng năng lượng (E)  : {total_energy:.2f}")

# So sánh 2 phân đoạn có đặc tính âm thanh khác nhau:
# Đoạn 1: Đoạn hòa âm cao trào (1.0s -> 2.0s)
# Đoạn 2: Đoạn khoảng lặng/nghỉ cuối tệp (8.5s -> 9.5s)
seg1 = x[int(1.0 * Fs):int(2.0 * Fs)]
seg2 = x[int(8.5 * Fs):int(9.5 * Fs)]

peak_seg1, rms_seg1 = np.max(np.abs(seg1)), np.sqrt(np.mean(seg1**2))
peak_seg2, rms_seg2 = np.max(np.abs(seg2)), np.sqrt(np.mean(seg2**2))

print(f"\n  [So sánh 2 phân đoạn đặc trưng]:")
print(f"    - Đoạn 1 (Hòa âm cao trào 1.0s-2.0s): Peak = {peak_seg1:.4f}, RMS = {rms_seg1:.4f} ({20*np.log10(rms_seg1):.2f} dBFS)")
print(f"    - Đoạn 2 (Khoảng lặng tĩnh 8.5s-9.5s): Peak = {peak_seg2:.4f}, RMS = {rms_seg2:.4f} ({20*np.log10(rms_seg2):.2f} dBFS)")
print(f"    -> Nhận xét: Đoạn 1 có năng lượng RMS gấp ~{rms_seg1/rms_seg2:.1f} lần Đoạn 2, biên độ dao động mạnh mẽ,")
print(f"       trong khi Đoạn 2 chỉ chứa nhiễu nền vi mô, không có hoạt động âm sắc nổi bật.")

# Vẽ đồ thị Waveform tổng hợp
time_axis = np.linspace(0, duration, len(x))
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Đồ thị 1: Toàn bộ tín hiệu
axes[0].plot(time_axis, x, color='#1f77b4', lw=0.6)
axes[0].axvspan(1.0, 2.0, color='orange', alpha=0.25, label='Đoạn 1: Hòa âm năng lượng cao')
axes[0].axvspan(8.5, 9.5, color='green', alpha=0.25, label='Đoạn 2: Khoảng lặng tĩnh')
axes[0].set_title(f"Waveform toàn bộ tệp âm thanh (Fs = {Fs} Hz, Duration = {duration:.2f}s, Peak = {peak_val:.3f})", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Thời gian (giây)")
axes[0].set_ylabel("Biên độ chuẩn hóa")
axes[0].set_ylim(-1.05, 1.05)
axes[0].legend(loc='upper right')

# Đồ thị 2: Zoom vào đoạn 0.5s đặc trưng (từ 1.0s đến 1.5s)
zoom_t = np.linspace(1.0, 1.5, int(0.5 * Fs))
zoom_sig = x[int(1.0 * Fs):int(1.5 * Fs)]
axes[1].plot(zoom_t, zoom_sig, color='#d62728', lw=1.0)
axes[1].set_title("Phân đoạn phóng to 0.5 giây (1.0s - 1.5s) thể hiện cấu trúc dao động chu kỳ của hòa âm", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Thời gian (giây)")
axes[1].set_ylabel("Biên độ")
axes[1].set_ylim(-1.05, 1.05)

# Đồ thị 3: So sánh trực tiếp Đoạn 1 vs Đoạn 2
t_local = np.linspace(0, 1.0, len(seg1))
axes[2].plot(t_local, seg1, color='orange', lw=0.9, label=f'Đoạn 1 (RMS = {rms_seg1:.3f})')
axes[2].plot(t_local, seg2, color='green', lw=0.9, alpha=0.8, label=f'Đoạn 2 (RMS = {rms_seg2:.3f})')
axes[2].set_title("So sánh trực quan hai phân đoạn miền thời gian: Đoạn âm thanh thực vs Khoảng lặng", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Thời gian cục bộ (giây)")
axes[2].set_ylabel("Biên độ")
axes[2].legend(loc='upper right')

plt.tight_layout()
waveform_fig_path = os.path.join(FIGURES_DIR, "waveform.png")
plt.savefig(waveform_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ miền thời gian: {waveform_fig_path}")


# =============================================================================
# KHỐI C: PHÂN TÍCH MIỀN TẦN SỐ BẰNG FFT (FREQUENCY-DOMAIN ANALYSIS)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI C] PHÂN TÍCH MIỀN TẦN SỐ BẰNG FFT")
print("=" * 80)

"""
Giải thích nguyên lý Khối C:
1. Bản chất toán học DFT (Discrete Fourier Transform):
   X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2*pi * k * n / N)
   - Thuật toán FFT (Fast Fourier Transform) tính DFT với độ phức tạp O(N log N).
   - Với tín hiệu thực (real-valued audio), phổ có tính đối xứng liên hợp Hermite: X[N-k] = X*[k].
     Do đó chỉ cần dùng hàm `np.fft.rfft` để lấy một nửa phổ từ 0 đến Nyquist (Fs/2).
2. Cửa sổ hóa (Windowing):
   - Khi cắt 1 đoạn ngắn, biên bị gián đoạn đột ngột (nhân cửa sổ chữ nhật ẩn), gây hiện tượng
     rò rỉ phổ (spectral leakage).
   - Nhân với cửa sổ Hamming w[n] = 0.54 - 0.46*cos(2*pi*n/(L-1)) làm mượt hai mép về 0,
     giảm thiểu việc năng lượng các vạch phổ tràn sang các dải tần lân cận.
3. Quy đổi trục tần số và Khoảng cách Bin (Frequency Bin Spacing):
   - Tần số tương ứng của bin thứ k: f_k = k * (Fs / NFFT).
   - Khoảng cách bin: delta_f = Fs / NFFT.
   - Thí nghiệm với 2 mức NFFT:
     * NFFT_small = 2048: delta_f = 44100 / 2048 = 21.53 Hz.
     * NFFT_large = 65536: delta_f = 44100 / 65536 = 0.673 Hz.
   - PHÂN BIỆT RÕ: Tăng NFFT bằng kỹ thuật Zero-Padding chỉ làm dải phổ dày đặc hơn (nội suy phổ sinc),
     nhưng KHÔNG LÀM TĂNG độ phân giải vật lý thực sự (True Resolution), vì độ phân giải vật lý
     bị giới hạn bởi chiều dài thời gian hữu hạn của cửa sổ phân tích (delta_f_true ~ 1 / T_window).
"""

# Chọn đoạn tín hiệu ổn định 1.0s (từ 1.0s đến 2.0s)
fft_seg = x[int(1.0 * Fs):int(2.0 * Fs)]
N_seg = len(fft_seg)
window_hamming = np.hamming(N_seg)
x_windowed = fft_seg * window_hamming

# 1. Tính FFT với NFFT = 65536 (độ phân giải bin cao)
NFFT_large = 65536
X_fft = np.fft.rfft(x_windowed, n=NFFT_large)
freqs_large = np.fft.rfftfreq(NFFT_large, d=1.0/Fs)
mag_linear = np.abs(X_fft)
# Chuẩn hóa về dB (dB relative to peak max = 0 dB)
mag_db = 20 * np.log10(np.maximum(mag_linear, 1e-12))
mag_db = mag_db - np.max(mag_db)  # Chuẩn hóa đỉnh cao nhất về 0 dB

delta_f_large = Fs / NFFT_large

# 2. Tìm ít nhất 3 đỉnh phổ nổi bật (Peak detection) trong dải tần [100 Hz, 4000 Hz]
search_idx = np.where((freqs_large >= 100) & (freqs_large <= 4000))[0]
peaks_indices, properties = signal.find_peaks(mag_db[search_idx], height=-25, distance=int(50/delta_f_large))
actual_peak_indices = search_idx[peaks_indices]

# Sắp xếp các đỉnh theo biên độ giảm dần và lấy top 5 đỉnh
top_indices = actual_peak_indices[np.argsort(mag_db[actual_peak_indices])[::-1][:5]]
top_indices = np.sort(top_indices)  # sắp xếp lại theo tần số tăng dần

print(f"  • Cấu hình FFT lớn      : NFFT = {NFFT_large}, Khoảng cách bin delta_f = {delta_f_large:.4f} Hz")
print(f"  • Các đỉnh phổ nổi bật phát hiện được:")
for rank, idx in enumerate(top_indices, 1):
    f_peak = freqs_large[idx]
    amp_peak = mag_db[idx]
    print(f"    - Đỉnh {rank}: Tần số = {f_peak:8.2f} Hz, Biên độ tương đối = {amp_peak:6.2f} dB")

# 3. Tính thêm FFT với NFFT nhỏ = 2048 để so sánh
NFFT_small = 2048
X_small = np.fft.rfft(x_windowed[:NFFT_small] * np.hamming(NFFT_small), n=NFFT_small)
freqs_small = np.fft.rfftfreq(NFFT_small, d=1.0/Fs)
mag_db_small = 20 * np.log10(np.maximum(np.abs(X_small), 1e-12))
mag_db_small -= np.max(mag_db_small)
delta_f_small = Fs / NFFT_small

# Vẽ đồ thị phổ FFT
plt.figure(figsize=(12, 5))
plt.plot(freqs_large, mag_db, color='#1f77b4', lw=0.9, label=rf'Phổ FFT (Hamming, NFFT={NFFT_large}, $\Delta f$={delta_f_large:.2f} Hz)')
for idx in top_indices:
    plt.plot(freqs_large[idx], mag_db[idx], "r*", markersize=10)
    plt.annotate(f"{freqs_large[idx]:.1f} Hz\n({mag_db[idx]:.1f} dB)",
                 xy=(freqs_large[idx], mag_db[idx]),
                 xytext=(freqs_large[idx] + 30, mag_db[idx] + 3),
                 fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
plt.title(f"Phổ biên độ FFT (Magnitude Spectrum theo dB) trên phân đoạn ổn định 1.0s - 2.0s", fontsize=11, fontweight='bold')
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ tương đối (dB)")
plt.xlim(0, 5000)
plt.ylim(-75, 5)
plt.legend(loc='upper right')
plt.tight_layout()
fft_fig_path = os.path.join(FIGURES_DIR, "fft_spectrum.png")
plt.savefig(fft_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ phổ FFT: {fft_fig_path}")

# Vẽ đồ thị so sánh NFFT nhỏ vs NFFT lớn
plt.figure(figsize=(12, 5))
plt.plot(freqs_small, mag_db_small, 'o-', color='orange', markersize=3, lw=1.0, alpha=0.8,
         label=rf'NFFT nhỏ = {NFFT_small} ($\Delta f$ = {delta_f_small:.2f} Hz - Lưới thưa, ít điểm nội suy)')
plt.plot(freqs_large, mag_db, color='#1f77b4', lw=0.8, alpha=0.9,
         label=rf'NFFT lớn = {NFFT_large} ($\Delta f$ = {delta_f_large:.2f} Hz - Lưới dày đặc, nội suy mượt mà)')
plt.title("So sánh độ phân giải lưới tần số (Frequency-Bin Spacing) giữa hai mức NFFT", fontsize=11, fontweight='bold')
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ tương đối (dB)")
plt.xlim(0, 2000)
plt.ylim(-65, 5)
plt.legend(loc='upper right')
plt.tight_layout()
fft_comp_fig_path = os.path.join(FIGURES_DIR, "fft_nfft_comparison.png")
plt.savefig(fft_comp_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ so sánh NFFT: {fft_comp_fig_path}")


# =============================================================================
# KHỐI D: STFT VÀ SPECTROGRAM (TIME-FREQUENCY ANALYSIS)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI D] PHÂN TÍCH THỜI GIAN - TẦN SỐ (STFT & SPECTROGRAM)")
print("=" * 80)

"""
Giải thích nguyên lý Khối D:
1. Tại sao cần STFT (Short-Time Fourier Transform)?
   - Tín hiệu âm thanh thực tế (tiếng nói, âm nhạc) là tín hiệu bất định / phi dừng (non-stationary),
     nghĩa là nội dung tần số thay đổi liên tục theo thời gian.
   - Phép biến đổi FFT toàn bài chỉ cho biết "những tần số nào có mặt" chứ không thể biết
     "tần số đó xuất hiện vào thời điểm nào".
   - STFT giải quyết bằng cách trượt một cửa sổ ngắn w[n] dọc theo trục thời gian:
     X[m, k] = sum_n x[n] * w[n - m*H] * exp(-j * 2*pi * k * n / NFFT)
     Trong đó:
       * L: Frame length (độ dài khung, ví dụ 25ms = 1102 mẫu ở Fs=44.1kHz).
       * H: Hop size (bước trượt giữa các khung liên tiếp, ví dụ 10ms = 441 mẫu).
       * Overlap: L - H (ví dụ 1102 - 441 = 661 mẫu, overlap ~ 60%).
2. Đánh đổi thời gian - tần số (Heisenberg-Gabor Trade-off):
   - Delta_t * Delta_f >= 1 / (4*pi).
   - Frame ngắn (10 ms): Độ phân giải thời gian rất cao (xác định chính xác từng xung nhịp gõ,
     transient), nhưng độ phân giải tần số kém (các dải tần bị bè to, nhòe).
   - Frame dài (50 ms): Độ phân giải tần số cao (tách biệt rõ từng vạch họa âm harmonic),
     nhưng độ phân giải thời gian bị làm mờ (các sự kiện gõ nhanh bị kéo dài).
"""

# Cấu hình chuẩn theo khuyến nghị của đề bài: L = 25ms, H = 10ms
L_standard = int(round(0.025 * Fs))  # ~1102 mẫu
H_standard = int(round(0.010 * Fs))  # ~441 mẫu
NFFT_stft = 4096  # Chọn 4096 để luôn >= nperseg cho các khung lên tới 50ms (2205 mẫu)

f_std, t_std, Sxx_std = signal.spectrogram(
    x, fs=Fs, window='hamming',
    nperseg=L_standard, noverlap=L_standard - H_standard,
    nfft=NFFT_stft, mode='magnitude'
)
Sxx_db_std = 20 * np.log10(np.maximum(Sxx_std, 1e-6))
vmin, vmax = np.percentile(Sxx_db_std, 15), np.max(Sxx_db_std)

f_mask_std = f_std <= 6000
plt.figure(figsize=(12, 5))
mesh = plt.pcolormesh(t_std, f_std[f_mask_std], Sxx_db_std[f_mask_std, :], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
plt.colorbar(mesh, label='Biên độ phổ (dB)')
plt.title(f"Spectrogram chuẩn (Khung L = 25 ms [{L_standard} mẫu], Bước nhảy H = 10 ms [{H_standard} mẫu], Overlap = 60%)",
          fontsize=11, fontweight='bold')
plt.xlabel("Thời gian (giây)")
plt.ylabel("Tần số (Hz)")
plt.ylim(0, 6000)
plt.tight_layout()
spec_std_fig_path = os.path.join(FIGURES_DIR, "spectrogram_standard.png")
plt.savefig(spec_std_fig_path, dpi=200)
plt.close()
print(f"  -> Đã lưu Spectrogram cấu hình chuẩn: {spec_std_fig_path}")

# Thí nghiệm so sánh 3 cấu hình Frame Length: 10 ms, 25 ms, 50 ms
frame_configs = [
    ("10 ms (Phân giải thời gian cao)", 0.010),
    ("25 ms (Chuẩn cân bằng)", 0.025),
    ("50 ms (Phân giải tần số cao)", 0.050)
]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
for idx, (label, frame_sec) in enumerate(frame_configs):
    L_cur = int(round(frame_sec * Fs))
    H_cur = int(round(0.010 * Fs))  # Hop size 10ms
    f_cur, t_cur, Sxx_cur = signal.spectrogram(
        x, fs=Fs, window='hamming',
        nperseg=L_cur, noverlap=L_cur - H_cur,
        nfft=NFFT_stft, mode='magnitude'
    )
    Sxx_db_cur = 20 * np.log10(np.maximum(Sxx_cur, 1e-6))
    f_mask = f_cur <= 6000
    im = axes[idx].pcolormesh(t_cur, f_cur[f_mask], Sxx_db_cur[f_mask, :], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[idx].set_title(f"Cấu hình {label} - Chiều dài L = {L_cur} mẫu ({frame_sec*1000:.0f} ms)", fontsize=10, fontweight='bold')
    axes[idx].set_ylabel("Tần số (Hz)")
    axes[idx].set_ylim(0, 5000)

axes[2].set_xlabel("Thời gian (giây)")
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Biên độ phổ (dB)')
spec_tradeoff_fig_path = os.path.join(FIGURES_DIR, "spectrogram_tradeoff.png")
plt.savefig(spec_tradeoff_fig_path, dpi=200)
plt.close()
print(f"  -> Đã lưu biểu đồ so sánh trade-off độ dài khung: {spec_tradeoff_fig_path}")


# =============================================================================
# KHỐI E: THÍ NGHIỆM CỬA SỔ (WINDOWING EXPERIMENT)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI E] THÍ NGHIỆM SO SÁNH CỬA SỔ (RECTANGULAR VS HAMMING)")
print("=" * 80)

"""
Giải thích nguyên lý Khối E:
1. Cắt tín hiệu liên tục thành hữu hạn tương đương với việc nhân tín hiệu với một cửa sổ w[n]:
   - Cửa sổ Chữ nhật (Rectangular Window):
     w[n] = 1 với mọi 0 <= n <= L-1.
     Đặc tính: Búp sóng chính (Main-lobe) rất hẹp (~ 4*pi / L), nhưng búp sóng phụ (Side-lobes)
     chỉ suy hao khoảng -13 dB so với đỉnh búp chính. Do đó năng lượng bị tràn sang các tần số xa
     rất mạnh mẽ -> Hiện tượng RÒ RỈ PHỔ (Spectral Leakage) nghiêm trọng.
   - Cửa sổ Hamming:
     w[n] = 0.54 - 0.46 * cos(2*pi*n / (L - 1)).
     Đặc tính: Búp sóng phụ bị dập tắt sâu xuống xấp xỉ -41 dB (giảm rò rỉ phổ cực kỳ tốt).
     Tuy nhiên cái giá phải trả là búp sóng chính bị bè rộng gấp đôi (~ 8*pi / L),
     khiến hai vạch phổ đứng rất sát nhau có thể bị hòa lẫn vào một đỉnh duy nhất.
2. Để so sánh công bằng: Phải giữ nguyên cùng một đoạn dữ liệu mẫu và cùng số điểm NFFT.
"""

frame_test = x[int(1.2 * Fs):int(1.2 * Fs + L_standard)]
N_win = len(frame_test)

w_rect = np.ones(N_win)
w_hamm = np.hamming(N_win)

NFFT_win = 8192
X_rect = np.fft.rfft(frame_test * w_rect, n=NFFT_win)
X_hamm = np.fft.rfft(frame_test * w_hamm, n=NFFT_win)
freqs_win = np.fft.rfftfreq(NFFT_win, d=1.0/Fs)

mag_rect_db = 20 * np.log10(np.maximum(np.abs(X_rect), 1e-12))
mag_hamm_db = 20 * np.log10(np.maximum(np.abs(X_hamm), 1e-12))
# Chuẩn hóa về đỉnh 0 dB để so sánh hình dạng búp sóng
mag_rect_db -= np.max(mag_rect_db)
mag_hamm_db -= np.max(mag_hamm_db)

plt.figure(figsize=(12, 5))
plt.plot(freqs_win, mag_rect_db, color='red', lw=0.9, alpha=0.75,
         label='Cửa sổ Chữ nhật (Rectangular) - Main-lobe nhọn, Side-lobes cao (~ -13 dB) gây rò rỉ phổ lớn')
plt.plot(freqs_win, mag_hamm_db, color='#1f77b4', lw=1.2,
         label='Cửa sổ Hamming - Main-lobe rộng hơn, Side-lobes thấp sâu (~ -41 dB) triệt tiêu rò rỉ phổ')
plt.title(f"So sánh phổ logarit trên cùng một khung tín hiệu ({L_standard} mẫu, NFFT={NFFT_win})", fontsize=11, fontweight='bold')
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ chuẩn hóa (dB)")
plt.xlim(0, 3000)
plt.ylim(-70, 5)
plt.legend(loc='upper right')
plt.tight_layout()
win_fig_path = os.path.join(FIGURES_DIR, "window_comparison.png")
plt.savefig(win_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ so sánh cửa sổ: {win_fig_path}")


# =============================================================================
# KHỐI F: THIẾT KẾ BỘ LỌC SỐ FIR VÀ LỌC TÍN HIỆU (DIGITAL FILTERING)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI F] THIẾT KẾ BỘ LỌC SỐ FIR VÀ LỌC TÍN HIỆU")
print("=" * 80)

"""
Giải thích nguyên lý Khối F:
1. Bộ lọc FIR (Finite Impulse Response - Đáp ứng xung hữu hạn):
   y[n] = sum_{r=0}^{M} b_r * x[n - r]
   - M là bậc bộ lọc, số hệ số numtaps = M + 1 (ví dụ numtaps = 201).
   - Thiết kế bằng phương pháp cửa sổ (Window Method) qua hàm `scipy.signal.firwin`:
     Lấy biến đổi Fourier ngược của đáp ứng tần số lý tưởng (dạng hàm Sinc) rồi nhân với cửa sổ Hamming.
   - Ưu điểm tuyệt đối của FIR đối xứng: Pha tuyến tính lý tưởng (Linear Phase), không gây méo pha
     (phase distortion), giữ nguyên hình dạng xung của tín hiệu.
2. Độ trễ nhóm tuyến tính (Group Delay):
   - Do tính đối xứng của bộ lọc FIR bậc chẵn, mọi thành phần tần số đều bị trễ một khoảng thời gian
     như nhau:
     tau = (numtaps - 1) / 2 mẫu.
     Với numtaps = 201, Fs = 44100 Hz:
     tau = (201 - 1) / 2 = 100 mẫu -> tau_ms = (100 / 44100) * 1000 = 2.27 ms.
   - Ý nghĩa: 2.27 ms là độ trễ cực nhỏ so với ngưỡng trễ nhận thức của thính giác người (~10-20 ms),
     do đó hoàn toàn đáp ứng tốt cho các hệ thống thời gian thực (Real-time).
3. Thiết kế:
   - 01 Bộ lọc Thông thấp (Low-pass Filter - LPF): fc = 2000 Hz.
   - 01 Bộ lọc Thông cao (High-pass Filter - HPF): fc = 3000 Hz.
   - 01 Bộ lọc Thông dải (Band-pass Filter - BPF): [1000 Hz, 3000 Hz].
"""

numtaps = 201
group_delay_samples = (numtaps - 1) // 2
group_delay_ms = (group_delay_samples / Fs) * 1000.0

print(f"  • Số hệ số lọc (numtaps): {numtaps} taps (Bậc M = {numtaps - 1})")
print(f"  • Độ trễ nhóm cố định   : {group_delay_samples} mẫu ({group_delay_ms:.2f} ms)")

# 1. Bộ lọc Thông thấp (Low-pass) fc = 2000 Hz
b_lpf = signal.firwin(numtaps=numtaps, cutoff=2000, fs=Fs, window='hamming', pass_zero='lowpass')
# 2. Bộ lọc Thông cao (High-pass) fc = 3000 Hz
b_hpf = signal.firwin(numtaps=numtaps, cutoff=3000, fs=Fs, window='hamming', pass_zero='highpass')
# 3. Bộ lọc Thông dải (Band-pass) [1000 Hz, 3000 Hz]
b_bpf = signal.firwin(numtaps=numtaps, cutoff=[1000, 3000], fs=Fs, window='hamming', pass_zero='bandpass')

# Tính đáp ứng tần số H(f) bằng signal.freqz
w_lpf, H_lpf = signal.freqz(b_lpf, worN=4096, fs=Fs)
w_hpf, H_hpf = signal.freqz(b_hpf, worN=4096, fs=Fs)
w_bpf, H_bpf = signal.freqz(b_bpf, worN=4096, fs=Fs)

# Vẽ biểu đồ đáp ứng tần số |H(f)| theo dB
plt.figure(figsize=(12, 5))
plt.plot(w_lpf, 20 * np.log10(np.maximum(np.abs(H_lpf), 1e-6)), color='#1f77b4', lw=1.5, label='FIR Low-pass (Cutoff = 2000 Hz)')
plt.plot(w_hpf, 20 * np.log10(np.maximum(np.abs(H_hpf), 1e-6)), color='#d62728', lw=1.5, label='FIR High-pass (Cutoff = 3000 Hz)')
plt.plot(w_bpf, 20 * np.log10(np.maximum(np.abs(H_bpf), 1e-6)), color='#2ca02c', lw=1.5, label='FIR Band-pass (1000 - 3000 Hz)')
plt.axhline(-3, color='black', linestyle='--', lw=0.8, label='Ngưỡng suy hao -3 dB (Cutoff level)')
plt.title(f"Đáp ứng tần số |H(f)| của các bộ lọc số FIR ({numtaps} taps, Hamming window, Fs = {Fs} Hz)", fontsize=11, fontweight='bold')
plt.xlabel("Tần số (Hz)")
plt.ylabel("Độ lợi biên độ |H(f)| (dB)")
plt.xlim(0, 6000)
plt.ylim(-80, 5)
plt.legend(loc='upper right')
plt.tight_layout()
filter_resp_fig_path = os.path.join(FIGURES_DIR, "filter_response.png")
plt.savefig(filter_resp_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ đáp ứng tần số bộ lọc: {filter_resp_fig_path}")

# Áp dụng bộ lọc lên tín hiệu x bằng signal.lfilter (tích chập trong miền thời gian)
y_lpf = signal.lfilter(b_lpf, [1.0], x)
y_hpf = signal.lfilter(b_hpf, [1.0], x)
y_bpf = signal.lfilter(b_bpf, [1.0], x)

# Xuất các file âm thanh sau lọc vào thư mục audio/
def export_wav(filename, signal_data, sample_rate):
    # Chuẩn hóa an toàn về int16 để ghi file wav
    sig_norm = np.clip(signal_data, -1.0, 1.0)
    wavfile.write(filename, sample_rate, np.int16(sig_norm * 32767))

export_wav(os.path.join(AUDIO_DIR, "filtered_lpf_2k.wav"), y_lpf, Fs)
export_wav(os.path.join(AUDIO_DIR, "filtered_hpf_3k.wav"), y_hpf, Fs)
export_wav(os.path.join(AUDIO_DIR, "filtered_bpf_1k_3k.wav"), y_bpf, Fs)
print(f"  -> Đã xuất 3 file âm thanh sau lọc vào: {AUDIO_DIR}")

# Vẽ đồ thị so sánh phổ trước và sau khi lọc Low-pass trên cùng một đoạn
N_test_filt = int(1.0 * Fs)
seg_orig = x[int(1.0 * Fs):int(2.0 * Fs)] * np.hamming(N_test_filt)
seg_filtered = y_lpf[int(1.0 * Fs):int(2.0 * Fs)] * np.hamming(N_test_filt)

X_orig = np.fft.rfft(seg_orig, n=16384)
X_filt = np.fft.rfft(seg_filtered, n=16384)
f_filt_axis = np.fft.rfftfreq(16384, d=1.0/Fs)

plt.figure(figsize=(12, 5))
plt.plot(f_filt_axis, 20 * np.log10(np.maximum(np.abs(X_orig), 1e-12)), color='#1f77b4', lw=0.9, alpha=0.7, label='Phổ trước khi lọc (Tín hiệu gốc)')
plt.plot(f_filt_axis, 20 * np.log10(np.maximum(np.abs(X_filt), 1e-12)), color='#d62728', lw=1.1, label='Phổ sau khi qua Low-pass 2000 Hz')
plt.axvline(2000, color='black', linestyle=':', lw=1.2, label='Tần số cắt fc = 2000 Hz')
plt.title("So sánh phổ biên độ trước và sau khi lọc Low-pass FIR (Minh họa triệt tiêu dải cao > 2000 Hz)", fontsize=11, fontweight='bold')
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ (dB)")
plt.xlim(0, 6000)
plt.ylim(-65, 30)
plt.legend(loc='upper right')
plt.tight_layout()
filter_comp_fig_path = os.path.join(FIGURES_DIR, "filter_spectrum_before_after.png")
plt.savefig(filter_comp_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ so sánh phổ trước/sau lọc: {filter_comp_fig_path}")


# =============================================================================
# KHỐI G: LƯỢNG TỬ HÓA, RESAMPLING VÀ MÃ HÓA (QUANTIZATION & CODING)
# =============================================================================
print("\n" + "=" * 80)
print("[KHỐI G] LƯỢNG TỬ HÓA, RESAMPLING VÀ MÃ HÓA ÂM THANH")
print("=" * 80)

"""
Giải thích nguyên lý Khối G:
1. Lượng tử hóa đều B-bit (Uniform Quantization):
   - Với B bit, không gian lượng tử có L = 2^B mức.
   - Đối xứng trên khoảng [-X_max, X_max] với X_max = 1.0:
     Bước lượng tử Delta = 2 * X_max / (2^B).
   - Tín hiệu sau lượng tử hóa x_q[n] được làm tròn về mức gần nhất.
   - Sai số lượng tử: e[n] = x_q[n] - x[n], giả thiết phân bố đều trong [-Delta/2, Delta/2].
   - Tỷ số tín hiệu trên nhiễu lượng tử thực nghiệm (Measured SNR):
     SNR_measured = 10 * log10( sum(x[n]^2) / sum((x_q[n] - x[n])^2) ) (dB).
   - Công thức lý thuyết Rabiner-Schafer:
     SNR_theoretical = 6.02 * B + 4.77 - 20 * log10(X_max / sigma_x) (dB).
     Trong đó sigma_x là độ lệch chuẩn (RMS) của tín hiệu.
2. Resampling (Lấy mẫu lại):
   - Khi hạ tần số lấy mẫu (downsampling) từ Fs=44.1kHz xuống Fs_new (ví dụ 16kHz hoặc 8kHz):
     Tần số Nyquist mới là Fs_new / 2 (tương ứng 8kHz và 4kHz).
   - Theo định lý Shannon-Nyquist, BẮT BUỘC phải dùng bộ lọc Low-pass chống chồng phổ (Anti-aliasing Filter)
     để triệt tiêu toàn bộ thành phần f > Fs_new / 2 trước khi rút bớt mẫu.
     Hàm `signal.resample_poly` của SciPy tự động thực hiện phép lọc pha tuyến tính này.
3. Tốc độ bit và Tỷ số nén (Coding & Compression):
   - Tốc độ bit chuẩn PCM: R_PCM = Fs * B * C (bps)
     Với CD-quality: 44100 * 16 * 2 = 1,411,200 bps = 1411.2 kbps.
   - So sánh với nén MP3 lossy: MP3 256 kbps hoặc MP3 128 kbps.
   - Compression Ratio = R_PCM / R_compressed.
   - Saving (%) = (1 - R_compressed / R_PCM) * 100%.
"""

def uniform_quantize(signal_in, B):
    """
    Hàm lượng tử hóa đều B-bit trên miền [-1.0, 1.0].
    qmax = 2^(B - 1) - 1.
    """
    qmax = 2**(B - 1) - 1
    sig_clipped = np.clip(signal_in, -1.0, 1.0)
    sig_quantized = np.round(sig_clipped * qmax) / qmax
    return sig_quantized

def calculate_snr(orig, quantized):
    """Tính tỷ số tín hiệu trên nhiễu (SNR) theo dB."""
    err = quantized - orig
    signal_pwr = np.sum(orig**2)
    noise_pwr = np.sum(err**2)
    if noise_pwr == 0:
        return float('inf')
    return 10.0 * np.log10(signal_pwr / noise_pwr)

bits_list = [4, 6, 8, 12, 16]
snr_measured = []
snr_theory = []

sigma_x = np.std(x)
X_max = 1.0
crest_factor_term = 20 * np.log10(X_max / sigma_x)

print(f"  • Độ lệch chuẩn (RMS) sigma_x = {sigma_x:.4f}")
print(f"  • Hệ số đỉnh 20*log10(X_max / sigma_x) = {crest_factor_term:.2f} dB")
print(f"\n  [Bảng kết quả Lượng tử hóa thực nghiệm vs Lý thuyết]:")
print(f"  {'Bit (B)':<8} | {'SNR Đo được (dB)':<18} | {'SNR Lý thuyết (dB)':<20} | {'Sai lệch (dB)':<12}")
print("  " + "-" * 65)

for B in bits_list:
    xq = uniform_quantize(x, B)
    meas = calculate_snr(x, xq)
    theo = 6.02 * B + 4.77 - crest_factor_term
    snr_measured.append(meas)
    snr_theory.append(theo)
    print(f"  {B:<8} | {meas:<18.2f} | {theo:<20.2f} | {abs(meas - theo):<12.2f}")
    
    # Xuất audio lượng tử hóa phục vụ nghe thử ở các mức quan trọng
    if B in [4, 8, 16]:
        export_wav(os.path.join(AUDIO_DIR, f"quantized_{B}bit.wav"), xq, Fs)

# Vẽ đồ thị SNR thực nghiệm vs lý thuyết
plt.figure(figsize=(10, 5))
plt.plot(bits_list, snr_measured, 'o-', color='#1f77b4', lw=1.5, markersize=7, label='SNR Đo được thực nghiệm')
plt.plot(bits_list, snr_theory, 's--', color='#d62728', lw=1.2, markersize=6, label=r'SNR Lý thuyết ($6.02B + 4.77 - 20\log_{10}(X_{max}/\sigma_x)$)')
plt.title("Tỷ số tín hiệu trên nhiễu lượng tử hóa (SNR) theo số bit B", fontsize=11, fontweight='bold')
plt.xlabel("Số bit trên mẫu (B)")
plt.ylabel("SNR (dB)")
plt.xticks(bits_list)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.tight_layout()
quant_fig_path = os.path.join(FIGURES_DIR, "quantization_snr.png")
plt.savefig(quant_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ SNR lượng tử hóa: {quant_fig_path}")

# Thí nghiệm Resampling về 16 kHz và 8 kHz (sử dụng signal.resample chống chồng phổ)
def resample_audio(sig, orig_fs, target_fs):
    """
    Hạ tần số lấy mẫu (Resampling / Downsampling) sử dụng biến đổi Fourier (signal.resample).
    Phương pháp này tự động cắt tần số vượt quá Nyquist mới (target_fs / 2) bằng bộ lọc lý tưởng,
    ngăn ngừa triệt để hiện tượng chồng phổ (Anti-Aliasing) theo đúng định lý Shannon-Nyquist.
    """
    target_num_samples = int(round(len(sig) * target_fs / orig_fs))
    return signal.resample(sig, target_num_samples)

sig_16k = resample_audio(x, Fs, 16000)
sig_8k = resample_audio(x, Fs, 8000)

export_wav(os.path.join(AUDIO_DIR, "resampled_16k.wav"), sig_16k, 16000)
export_wav(os.path.join(AUDIO_DIR, "resampled_8k.wav"), sig_8k, 8000)
print(f"  -> Đã xuất 2 file audio sau Resampling về 16kHz và 8kHz vào: {AUDIO_DIR}")

# Tính toán Bit rate và Tỷ số nén MP3 vs PCM
R_pcm_stereo = Fs * 16 * 2  # 1,411,200 bps
size_pcm_mb = (R_pcm_stereo * duration) / (8 * 1024 * 1024)

codecs = [
    ("PCM 16-bit Stereo (Gốc)", 1411.2, size_pcm_mb),
    ("MP3 256 kbps (High Quality)", 256.0, (256000 * duration) / (8 * 1024 * 1024)),
    ("MP3 128 kbps (Standard)", 128.0, (128000 * duration) / (8 * 1024 * 1024)),
    ("Opus 64 kbps (Voice/Web)", 64.0, (64000 * duration) / (8 * 1024 * 1024))
]

print(f"\n  [So sánh Tốc độ bit và Dung lượng lưu trữ cho thời lượng {duration:.2f}s]:")
print(f"  {'Định dạng / Codec':<28} | {'Bitrate (kbps)':<15} | {'Dung lượng (MB)':<16} | {'Tỷ số nén':<12} | {'Tiết kiệm (%)':<14}")
print("  " + "-" * 90)

for name, br, sz in codecs:
    ratio = 1411.2 / br
    saving = (1.0 - br / 1411.2) * 100.0
    print(f"  {name:<28} | {br:<15.1f} | {sz:<16.3f} | {ratio:<12.2f}x | {saving:<14.1f}%")

# Vẽ đồ thị so sánh Bit rate
names = [c[0] for c in codecs]
bitrates = [c[1] for c in codecs]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

plt.figure(figsize=(10, 5))
bars = plt.bar(names, bitrates, color=colors, width=0.55, edgecolor='black', lw=0.8)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 25, f"{yval:.1f} kbps", ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.title(f"So sánh tốc độ bit (Bit Rate) giữa định dạng PCM nguyên bản và các chuẩn mã hóa nén", fontsize=11, fontweight='bold')
plt.ylabel("Tốc độ bit (kbps)")
plt.ylim(0, 1600)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
bitrate_fig_path = os.path.join(FIGURES_DIR, "bitrate_comparison.png")
plt.savefig(bitrate_fig_path, dpi=300)
plt.close()
print(f"  -> Đã lưu biểu đồ so sánh bitrate: {bitrate_fig_path}")

print("\n" + "=" * 80)
print("HOÀN TẤT TOÀN BỘ PIPELINE THỰC NGHIỆM LAB 1!")
print("Tất cả âm thanh đã được xuất vào: lab1/audio/")
print("Tất cả biểu đồ đã được xuất vào: lab1/figures/")
print("=" * 80)
