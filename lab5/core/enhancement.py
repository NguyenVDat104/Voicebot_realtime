"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
Mô-đun 1: Khử Nhiễu Tiếng Nói Trong Miền STFT (Speech Enhancement)
Học phần: CSE457 - Trường Đại học Thủy Lợi

Nội dung:
- Mô hình cộng nhiễu y[n] = s[n] + alpha * v[n] tại target SNR.
- Biến đổi STFT và ước lượng phổ công suất nhiễu Pn[k].
- Khử nhiễu bằng Spectral Subtraction với tham số over-subtraction beta và spectral floor gamma.
- Khử nhiễu bằng Wiener Filter tối ưu MMSE.
- Căn chỉnh pha/độ trễ và đo đạc SNR_in, SNR_out, Delta_SNR chuẩn xác.
===============================================================================
"""

import numpy as np
import librosa
from scipy import signal


def load16k(path):
    """Đọc file âm thanh về dạng mono 16 kHz, float64."""
    x, sr = librosa.load(path, sr=16000, mono=True)
    return x.astype(np.float64), sr


def power(x):
    """Tính công suất trung bình P_x = (1/N) sum(x^2)."""
    return float(np.mean(np.asarray(x, dtype=np.float64)**2) + 1e-12)


def rms(x):
    """Tính căn bậc hai công suất trung bình RMS(x)."""
    return float(np.sqrt(power(x)))


def add_noise_at_snr(s, v, snr_db):
    """
    Tạo tín hiệu tiếng nói bị nhiễu y[n] = s[n] + alpha * v[n] tại SNR mục tiêu (dB).
    alpha = sqrt( P_s / (P_v * 10^(snr_db / 10)) )
    """
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Đảm bảo v đủ độ dài bằng cách lặp (tile) hoặc cắt
    if len(v) < len(s):
        repeats = int(np.ceil(len(s) / len(v)))
        v = np.tile(v, repeats)
    v = v[:len(s)]

    p_s = power(s)
    p_v = power(v)
    alpha = np.sqrt(p_s / (p_v * (10.0 ** (snr_db / 10.0))))

    y = s + alpha * v
    return y, float(alpha)


def align_signal(ref, test, max_lag=400):
    """
    Căn chỉnh độ trễ thời gian nhỏ (do STFT/ISTFT hoặc filter delay)
    bằng tương quan chéo (cross-correlation) để đo SNR chuẩn xác nhất.
    """
    L = min(len(ref), len(test))
    ref_c = ref[:L]
    test_c = test[:L]

    corr = signal.correlate(test_c[:2000], ref_c[:2000], mode='full')
    lags = signal.correlation_lags(len(test_c[:2000]), len(ref_c[:2000]), mode='full')
    best_lag = lags[np.argmax(corr)]

    # Giới hạn lag trong khoảng [-max_lag, max_lag]
    if abs(best_lag) <= max_lag and best_lag != 0:
        if best_lag > 0:
            test_aligned = np.pad(test_c[best_lag:], (0, best_lag))
        else:
            test_aligned = np.pad(test_c, (-best_lag, 0))[:L]
        return ref_c, test_aligned
    return ref_c, test_c


def snr_db(ref, test, align=True):
    """
    Tính SNR (dB) giữa tín hiệu sạch tham chiếu (ref) và tín hiệu kiểm thử (test).
    SNR = 10 * log10( sum(ref^2) / sum((ref - test)^2) )
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    if align:
        ref_a, test_a = align_signal(ref, test)
    else:
        L = min(len(ref), len(test))
        ref_a, test_a = ref[:L], test[:L]

    err = test_a - ref_a
    ref_pow = np.sum(ref_a ** 2)
    err_pow = np.sum(err ** 2) + 1e-12
    return float(10.0 * np.log10(ref_pow / err_pow))


def stft_forward(y, sr=16000, n_fft=512, win_length=400, hop_length=160):
    """Thực hiện biến đổi STFT với cửa sổ Hann."""
    return librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window='hann'
    )


def istft_backward(Y_stft, target_len=None, sr=16000, hop_length=160, win_length=400):
    """Tái tạo waveform từ phổ STFT bằng ISTFT/Overlap-Add."""
    return librosa.istft(
        Y_stft, hop_length=hop_length, win_length=win_length,
        length=target_len
    )


def estimate_noise_psd(noise, n_fft=512, win_length=400, hop_length=160):
    """Ước lượng phổ công suất nhiễu (PSD) trung bình: P_n[k] = (1/M) sum |N[m,k]|^2."""
    N = librosa.stft(
        noise, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window='hann'
    )
    # Pn có shape (1 + n_fft//2, 1) để broadcast qua các khung thời gian
    Pn = np.mean(np.abs(N) ** 2, axis=1, keepdims=True)
    return Pn


def enhance_spectral_subtraction(y, noise_ref, sr=16000, beta=1.5, floor=0.02,
                                 n_fft=512, win_length=400, hop_length=160):
    """
    Khử nhiễu bằng Spectral Subtraction (Trừ phổ):
    |S_hat[m,k]|^2 = max( |Y[m,k]|^2 - beta * Pn[k], floor * Pn[k] )
    Giữ nguyên pha góc pha của Y: S_hat = |S_hat| * exp(j * angle(Y))
    """
    Y = stft_forward(y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    Pn = estimate_noise_psd(noise_ref, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    Py = np.abs(Y) ** 2

    # Trừ phổ có hệ số over-subtraction beta và spectral floor gamma
    Pss = np.maximum(Py - beta * Pn, floor * Pn)
    Sss = np.sqrt(Pss) * np.exp(1j * np.angle(Y))

    x_ss = istft_backward(Sss, target_len=len(y), sr=sr, hop_length=hop_length, win_length=win_length)
    return x_ss


def enhance_wiener(y, noise_ref, sr=16000, eps=1e-12,
                   n_fft=512, win_length=400, hop_length=160):
    """
    Khử nhiễu bằng bộ lọc Wiener tối ưu MMSE:
    Ps_hat = max(Py - Pn, 0.0)
    H[m,k] = Ps_hat / (Ps_hat + Pn + eps)
    S_wiener = H * Y
    """
    Y = stft_forward(y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    Pn = estimate_noise_psd(noise_ref, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    Py = np.abs(Y) ** 2

    Ps = np.maximum(Py - Pn, 0.0)
    H = Ps / (Ps + Pn + eps)
    Sw = H * Y

    x_w = istft_backward(Sw, target_len=len(y), sr=sr, hop_length=hop_length, win_length=win_length)
    return x_w, H


def enhance_both(y, noise_ref, clean_ref=None, sr=16000, beta=1.5, floor=0.02):
    """Tiện ích thực hiện đồng thời cả 2 phương pháp và tính toán SNR."""
    x_ss = enhance_spectral_subtraction(y, noise_ref, sr=sr, beta=beta, floor=floor)
    x_w, H = enhance_wiener(y, noise_ref, sr=sr)

    stats = {}
    if clean_ref is not None:
        snr_in = snr_db(clean_ref, y)
        snr_ss = snr_db(clean_ref, x_ss)
        snr_w = snr_db(clean_ref, x_w)
        stats = {
            "snr_in": snr_in,
            "snr_out_ss": snr_ss,
            "snr_out_wiener": snr_w,
            "delta_ss": snr_ss - snr_in,
            "delta_wiener": snr_w - snr_in
        }

    return x_ss, x_w, stats
