"""
===============================================================================
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 2
Công cụ Ghi âm Trực tiếp qua Microphone (Interactive Voice Recorder)

Tệp: record_dataset.py
Mục đích:
  - Cho phép người dùng trực tiếp sử dụng Microphone của máy tính để ghi âm
    bộ dữ liệu 5 từ tiếng Việt của chính mình: ["không", "một", "hai", "ba", "bốn"].
  - Tự động chuẩn hóa:
      + Đơn kênh (Mono), Sampling Rate Fs = 16,000 Hz, Chuẩn 16-bit PCM.
      + Hướng dẫn phát âm tách rời và giữ khoảng lặng 0.3s trước/sau từ nói.
      + Tự động lưu vào đúng cấu trúc thư mục: dataset/<từ>/<từ>_<stt>.wav.

Cách dùng:
  python record_dataset.py
===============================================================================
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000          # Tần số lấy mẫu chuẩn của Lab 2 (16 kHz)
DURATION = 2.5      # Thời lượng thu âm cho mỗi từ (2.5 giây đủ cho silence + từ đơn + silence)
VOCAB = ['khong', 'mot', 'hai', 'ba', 'bon']
VIET_MAP = {
    'khong': 'KHÔNG',
    'mot': 'MỘT',
    'hai': 'HAI',
    'ba': 'BA',
    'bon': 'BỐN'
}


def record_sample(word, idx, target_dir):
    """Ghi âm 1 mẫu phát âm cho 1 từ."""
    save_dir = os.path.join(target_dir, word)
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f"{word}_{idx:02d}.wav")

    print("\n" + "-" * 55)
    print(f" Chuẩn bị ghi âm từ: [{VIET_MAP[word]}] - Lần lặp {idx}/5")
    print(" LƯU Ý: Giữ im lặng 0.3s -> ĐỌC TỪ RÕ RÀNG -> Giữ im lặng 0.3s")
    input(" Nhấn [ENTER] khi đã sẵn sàng...")

    for count in range(3, 0, -1):
        print(f"   Bắt đầu trong: {count}...", end="\r", flush=True)
        time.sleep(1)

    print("   >>> ĐANG THU ÂM! HÃY NÓI TỪ: [" + VIET_MAP[word] + "] <<<")
    
    # Thu âm qua sounddevice
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    print("   ---> ĐÃ THU XONG!")

    audio = audio.flatten()
    max_val = np.max(np.abs(audio))

    # Cảnh báo nếu tín hiệu quá nhỏ hoặc quá lớn
    if max_val < 0.05:
        print("   [CẢNH BÁO] Âm lượng quá nhỏ! Có thể mic chưa bật hoặc nói quá bé.")
    elif max_val > 0.98:
        print("   [CẢNH BÁO] Âm lượng bị quá lớn (nguy cơ clipping). Hãy lùi xa mic một chút.")
    else:
        # Chuẩn hóa biên độ đỉnh về 0.90
        audio = audio / max_val * 0.90

    # Lưu file WAV 16-bit PCM
    sf.write(out_file, audio, FS, subtype='PCM_16')
    print(f"   [LƯU THÀNH CÔNG]: {out_file} ({len(audio)/FS:.2f}s, Peak: {max_val:.2f})")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")

    print("=" * 60)
    print(" CÔNG CỤ GHI ÂM DỮ LIỆU LAB 2 - NHẬN DẠNG TIẾNG NÓI")
    print(" Trường ĐH Thủy Lợi - CSE457")
    print(f" Thư mục lưu dữ liệu: {dataset_dir}")
    print("=" * 60)
    print("Chọn chế độ ghi âm:")
    print("  1. Ghi âm lần lượt toàn bộ 25 file (5 từ x 5 lần)")
    print("  2. Ghi âm một từ cụ thể")
    print("  3. Thoát")

    choice = input("\nNhập lựa chọn (1/2/3): ").strip()

    if choice == '1':
        for word in VOCAB:
            for i in range(1, 6):
                record_sample(word, i, dataset_dir)
        print("\n Hoàn tất ghi âm 25 file âm thanh thực tế!")

    elif choice == '2':
        print(f"Danh sách từ: {', '.join(VOCAB)}")
        w = input("Nhập từ muốn ghi âm (khong/mot/hai/ba/bon): ").strip().lower()
        if w not in VOCAB:
            print("Từ không hợp lệ!")
            return
        idx = input("Nhập số thứ tự lần lặp (1-5): ").strip()
        try:
            idx = int(idx)
            if 1 <= idx <= 5:
                record_sample(w, idx, dataset_dir)
            else:
                print("Chỉ được nhập từ 1 đến 5.")
        except ValueError:
            print("Số thứ tự không hợp lệ.")

    else:
        print("Đã thoát công cụ ghi âm.")


if __name__ == '__main__':
    main()
