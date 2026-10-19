"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 3
Công cụ Ghi âm Trực tiếp qua Microphone (Interactive Voice Recorder)
===============================================================================
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Vui lòng cài đặt: pip install sounddevice soundfile")
    sys.exit(1)

FS = 16000
DURATION = 2.5
VOCAB = ['mo', 'dong', 'len', 'xuong', 'dung']
VIET_MAP = {
    'mo': 'MỞ',
    'dong': 'ĐÓNG',
    'len': 'LÊN',
    'xuong': 'XUỐNG',
    'dung': 'DỪNG'
}


def record_sample(word, idx, target_dir):
    save_dir = Path(target_dir) / word
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{word}_{idx:02d}.wav"

    split_str = "TRAIN" if idx <= 6 else "TEST"
    print("\n" + "-" * 60)
    print(f" [{split_str}] Chuẩn bị ghi âm lệnh: [{VIET_MAP[word]}] - Mẫu {idx:02d}/08")
    print(" HƯỚNG DẪN: Giữ im lặng 0.3s -> NÓI RÕ RÀNG -> Giữ im lặng 0.3s")
    input(" Nhấn [ENTER] khi bạn đã sẵn sàng...")

    for count in range(3, 0, -1):
        print(f"   Bắt đầu trong: {count}...", end="\r", flush=True)
        time.sleep(1)

    print("   >>> ĐANG THU ÂM! HÃY NÓI: [" + VIET_MAP[word] + "] <<<")
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    print("   ---> ĐÃ THU XONG!")

    data = audio.flatten()
    max_val = np.max(np.abs(data))
    if max_val > 0.01:
        data = data / max_val * 0.90
    else:
        print("   [CẢNH BÁO] Âm lượng quá nhỏ, vui lòng kiểm tra lại Microphone!")

    sf.write(str(out_file), data, FS, subtype='PCM_16')
    print(f"   Đã lưu: {out_file} (16kHz PCM 16-bit)")


def main():
    base_dir = Path(__file__).resolve().parent
    train_dir = base_dir / "dataset" / "train"
    test_dir = base_dir / "dataset" / "test"

    print("=" * 60)
    print("CHƯƠNG TRÌNH THU ÂM BỘ DỮ LIỆU LAB 3 (5 LỆNH ĐIỀU KHIỂN)")
    print("Mỗi lệnh sẽ thu 8 lần: 6 lần đầu làm TRAIN, 2 lần sau làm TEST")
    print("=" * 60)

    for word in VOCAB:
        for idx in range(1, 9):
            target_dir = train_dir if idx <= 6 else test_dir
            record_sample(word, idx, target_dir)

    print("\n" + "=" * 60)
    print("HOÀN TẤT THU ÂM TOÀN BỘ 40 MẪU CHO 5 LỆNH TIẾNG VIỆT!")
    print("=" * 60)


if __name__ == '__main__':
    main()
