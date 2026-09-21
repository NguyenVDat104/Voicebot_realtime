"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 1
Mô-đun Tiện ích: Chuyển đổi định dạng âm thanh từ MP3 sang WAV (PCM)

Tệp: convert_mp3_to_wav.py
Mục đích:
  - Chuyển đổi các tệp nén MP3 (lossy) sang định dạng không nén WAV (Linear PCM).
  - Hỗ trợ đa nền tảng thư viện (tự động ưu tiên thư viện có sẵn trong môi trường):
      1. PyAV (`import av`): Đã tích hợp sẵn bộ giải mã libav/ffmpeg trong môi trường.
      2. PyDub (`from pydub import AudioSegment`): Chuẩn theo tài liệu hướng dẫn Lab 1.
      3. Librosa / SoundFile: Thư viện nghiên cứu âm thanh phổ biến.
  - Lưu ý lý thuyết (Mục 2.8 Lab 1):
    Chuyển đổi từ MP3 đã nén sang WAV chỉ thay đổi định dạng lưu trữ (container),
    KHÔNG THỂ phục hồi lại thông tin tần số cao đã bị mất do mã hóa lossy trước đó!

Cách sử dụng:
  Cách 1 (Dòng lệnh):
    python convert_mp3_to_wav.py duong_dan_file.mp3 [duong_dan_file.wav]

  Cách 2 (Gọi hàm trong mã nguồn Python):
    from convert_mp3_to_wav import convert_mp3_to_wav
    convert_mp3_to_wav("audio/sample.mp3", "audio/sample.wav")
===============================================================================
"""

import os
import sys
import numpy as np
from scipy.io import wavfile


def convert_via_pyav(mp3_path, wav_path):
    """
    Chuyển đổi MP3 sang WAV sử dụng PyAV (thư viện có sẵn trong môi trường chatbot).
    PyAV giải mã trực tiếp luồng bitstream nén MP3 thành các mẫu PCM 16-bit.
    """
    import av

    print(f"  [Động cơ: PyAV] Đang giải mã '{os.path.basename(mp3_path)}'...")
    container = av.open(mp3_path)
    stream = container.streams.audio[0]
    sample_rate = stream.rate

    audio_frames = []
    for frame in container.decode(stream):
        # frame.to_ndarray() trả về mảng shape (channels, samples)
        audio_frames.append(frame.to_ndarray())

    container.close()

    if not audio_frames:
        raise ValueError("Không tìm thấy dữ liệu âm thanh hợp lệ trong tệp MP3!")

    # Ghép các khung thời gian liên tiếp lại
    audio_data = np.concatenate(audio_frames, axis=1)

    # Nếu stereo/đa kênh, đổi trục (channels, samples) -> (samples, channels)
    if audio_data.shape[0] == 1:
        audio_out = audio_data[0]
    else:
        audio_out = audio_data.T

    # Đảm bảo dữ liệu xuất ra ở định dạng chuẩn 16-bit PCM ([-32768, 32767])
    if audio_out.dtype != np.int16:
        if np.issubdtype(audio_out.dtype, np.floating):
            # Nếu là số thực float trong [-1.0, 1.0]
            audio_out = (np.clip(audio_out, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_out = audio_out.astype(np.int16)

    # Ghi ra tệp WAV chuẩn PCM bằng scipy.io.wavfile
    wavfile.write(wav_path, sample_rate, audio_out)
    return sample_rate, audio_out.shape


def convert_via_pydub(mp3_path, wav_path):
    """
    Chuyển đổi MP3 sang WAV sử dụng PyDub (theo hướng dẫn Mục 4.1 trong Lab 1).
    """
    from pydub import AudioSegment

    print(f"  [Động cơ: PyDub] Đang giải mã '{os.path.basename(mp3_path)}'...")
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    audio.export(wav_path, format="wav")
    return audio.frame_rate, (len(audio.get_array_of_samples()), audio.channels)


def convert_via_librosa(mp3_path, wav_path):
    """
    Chuyển đổi MP3 sang WAV sử dụng Librosa / Soundfile.
    """
    import librosa
    import soundfile as sf

    print(f"  [Động cơ: Librosa] Đang giải mã '{os.path.basename(mp3_path)}'...")
    # sr=None để giữ nguyên tần số lấy mẫu gốc của tệp MP3
    y, sr = librosa.load(mp3_path, sr=None, mono=False)
    # Ghi ra định dạng WAV 16-bit PCM
    if y.ndim == 1:
        sf.write(wav_path, y, sr, subtype='PCM_16')
    else:
        sf.write(wav_path, y.T, sr, subtype='PCM_16')
    return sr, y.shape


def convert_mp3_to_wav(mp3_path, wav_path=None):
    """
    Hàm chuyển đổi tổng quát MP3 sang WAV.
    Tự động thử các thư viện khả dụng theo thứ tự ưu tiên:
    1. PyAV (tích hợp sẵn trong conda env chatbot)
    2. PyDub (khuyến nghị trong đề bài)
    3. Librosa / SoundFile
    """
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: '{mp3_path}'")

    if wav_path is None:
        # Nếu không chỉ định đường dẫn ra, tự động đổi đuôi .mp3 -> .wav
        base, _ = os.path.splitext(mp3_path)
        wav_path = base + ".wav"

    # Tạo thư mục đích nếu chưa có
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)

    print("-" * 70)
    print(f"BẮT ĐẦU CHUYỂN ĐỔI: {os.path.basename(mp3_path)} -> {os.path.basename(wav_path)}")
    print(f"  • Nguồn: {mp3_path}")
    print(f"  • Đích : {wav_path}")

    # Danh sách các hàm chuyển đổi thử nghiệm
    backends = [
        ("PyAV", convert_via_pyav),
        ("PyDub", convert_via_pydub),
        ("Librosa", convert_via_librosa)
    ]

    last_error = None
    for name, func in backends:
        try:
            sr, shape = func(mp3_path, wav_path)
            file_size = os.path.getsize(wav_path)
            print(f"  -> THÀNH CÔNG!")
            print(f"  • Tần số lấy mẫu (Fs) : {sr:,} Hz")
            print(f"  • Kích thước mảng số  : {shape}")
            print(f"  • Dung lượng file WAV : {file_size:,} bytes (~{file_size / (1024*1024):.2f} MB)")
            print("-" * 70)
            return wav_path
        except ImportError:
            continue
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Không thể chuyển đổi tệp MP3 sang WAV! Lỗi: {last_error}\n"
        "Vui lòng đảm bảo môi trường có cài đặt một trong các thư viện: `av`, `pydub` (kèm ffmpeg), hoặc `librosa`."
    )


if __name__ == "__main__":
    print("=" * 70)
    print("TIỆN ÍCH CHUYỂN ĐỔI ÂM THANH MP3 -> WAV (CSE457 LAB 1)")
    print("=" * 70)

    if len(sys.argv) >= 2:
        input_mp3 = sys.argv[1]
        output_wav = sys.argv[2] if len(sys.argv) >= 3 else None
        convert_mp3_to_wav(input_mp3, output_wav)
    else:
        # Tìm các tệp âm thanh nén trong thư mục audio/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_dir = os.path.join(current_dir, "audio")
        compressed_exts = (".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma")
        compressed_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(compressed_exts)] if os.path.exists(audio_dir) else []

        if compressed_files:
            sample_file = os.path.join(audio_dir, compressed_files[0])
            print(f"Phát hiện tệp âm thanh nén trong thư mục audio/: '{compressed_files[0]}'")
            convert_mp3_to_wav(sample_file)
        else:
            print("CÁCH SỬ DỤNG:")
            print("  python convert_mp3_to_wav.py <duong_dan_file.mp3> [duong_dan_file.wav]")
            print("\nVí dụ:")
            print('  python convert_mp3_to_wav.py "audio/Western Cowboy Texas Music.mp3"')
            print('  python convert_mp3_to_wav.py "audio/input.mp3" "audio/input.wav"')
