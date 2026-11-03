# CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 4
## TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH - TTS)
**Trường Đại học Thủy Lợi – Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo**  
**Sinh viên thực hiện:** Nguyễn Văn Đạt  

---

## 1. Cấu trúc Thư mục Dự án Lab 4
```text
Speech&Audio/lab4/
├── Lab 4.pdf                    <- Đề bài gốc Lab 4
├── PLAN.md                     <- Kế hoạch triển khai tổng thể chi tiết (Master Plan)
├── README.md                   <- Hướng dẫn thiết lập môi trường & sử dụng
├── requirements.txt            <- Danh mục thư viện Python cần thiết
├── generate_inventory.py       <- Script tự động sinh kho dữ liệu Inventory 48 WAV
├── record_inventory.py         <- Công cụ thu âm qua micro tương tác
├── hmm_tts_backend.py          <- Module HMM Parametric trajectory generation & vocoder
├── neural_tts_backend.py       <- Module Neural TTS Pretrained offline wrapper & latency profiler
├── run_lab4.py                 <- Pipeline tự động chạy toàn diện từ A -> G
├── Lab04.ipynb                 <- Jupyter Notebook tương tác trực quan 3 tiết thực hành
├── report_Lab04.md             <- Báo cáo học thuật 5-8 trang, giải chi tiết 10 câu hỏi
├── ratings.csv                 <- Kết quả đánh giá MOS / Listening test (72 lượt chấm)
├── lab4_data/
│   ├── units/                  <- 48 file WAV mono 16kHz của 16 từ vựng (3 candidates/từ)
│   ├── metadata.csv            <- Bảng thông số kỹ thuật (duration, rms, samples)
│   └── neural_cache/           <- Cache audio offline cho Neural TTS
├── audio_output/               <- Thư mục chứa toàn bộ sản phẩm âm thanh
│   ├── raw_concat.wav          <- Ghép trực tiếp baseline (xfade = 0ms)
│   ├── best_concat.wav         <- Sau Unit Selection DP + Cross-fade 15ms tối ưu
│   ├── hmm_tts.wav             <- Sinh từ tham số HMM
│   ├── neural_tts.wav          <- Sinh từ Pretrained Neural TTS
│   ├── xfade_05ms.wav / 30ms   <- Các file khảo sát cross-fade
│   └── prosody_*.wav           <- Các file khảo sát time-stretch & pitch-shift
└── figures/                    <- Bộ 6 hình đồ thị phân tích chất lượng cao
    ├── fig1_inventory_stats.png        <- Histogram duration và RMS của Inventory
    ├── fig2_spectrogram_boundaries.png <- Waveform & Spectrogram đánh dấu biên nối
    ├── fig3_unit_selection_matrix.png  <- Ma trận Join Cost & Lưới DP Trellis
    ├── fig4_prosody_modifications.png  <- Autocorrelation F0 và biến đổi Prosody
    ├── fig5_hmm_trajectories.png       <- Quỹ đạo tham số HMM & vai trò Delta-feature
    └── fig6_comparative_spectrogram.png<- So sánh đầu ra 3 phương pháp
```

---

## 2. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### 2.1 Cài đặt môi trường
```powershell
pip install numpy scipy matplotlib soundfile librosa pandas edge-tts sounddevice
```

### 2.2 Chạy toàn bộ Pipeline tự động
Để chạy tự động từ A đến G, sinh toàn bộ audio và hình ảnh:
```powershell
python run_lab4.py
```

### 2.3 Thu âm giọng thật (nếu cần nộp giọng sinh viên)
```powershell
python record_inventory.py
```

### 2.4 Chạy Jupyter Notebook tương tác
Khởi động Jupyter và mở file [Lab04.ipynb](file:///c:/Speech&Audio/lab4/Lab04.ipynb) để trải nghiệm giao diện âm thanh tương tác theo đúng phân bổ 3 tiết học.
