# CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
## SPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION
### Từ Khử Nhiễu Trong Miền STFT Đến Xác Định Người Nói Và Tách Hai Nguồn Nói Bằng ICA
**Trường Đại học Thủy Lợi – Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo**  
**Sinh viên thực hiện:** Nguyễn Văn Đạt  
**Mã sinh viên:** 2151062768  

---

## 1. Cấu trúc Thư mục Dự án Lab 5
```text
Speech&Audio/lab5/
├── Lab 5.pdf                     <- Đề bài và rubric gốc của Lab 5
├── PLAN.md                       <- Bản thiết kế kiến trúc và kế hoạch tổng thể
├── README.md                     <- Tài liệu hướng dẫn thiết lập môi trường & tái lập kết quả
├── requirements.txt              <- Danh mục các thư viện Python phụ thuộc
├── generate_dataset.py           <- Script tự động tổng hợp bộ dữ liệu chuẩn 16kHz
├── run_lab5.py                   <- Master Runner thực thi toàn diện Case Study và 8 thí nghiệm bắt buộc
├── make_notebook.py              <- Script sinh mã nguồn Jupyter Notebook Lab05.ipynb
├── Lab05.ipynb                   <- Notebook tương tác trực quan 3 tiết thực hành (đã chạy sẵn outputs)
├── report_Lab05.md               <- Báo cáo học thuật chi tiết 4–6 trang, giải trọn vẹn 12 câu hỏi lý thuyết
├── lab5_data/                    <- Bộ dữ liệu âm thanh chuẩn hóa:
│   ├── clean/                    <- 18 tệp WAV 16kHz Mono (spk01, spk02, spk03 x 4 enroll + 2 test)
│   ├── noise/                    <- fan.wav (tiếng quạt gió), cafeteria.wav (tiếng ồn quán ăn)
│   └── separation/               <- source_A.wav, source_B.wav (cùng độ dài 3.5s, RMS=0.08)
├── core/                         <- Các mô-đun lõi xử lý tín hiệu:
│   ├── __init__.py
│   ├── enhancement.py            <- STFT, ước lượng PSD nhiễu, Spectral Subtraction, Wiener filter, SNR
│   ├── speaker_rec.py            <- MFCC, Delta, 80-D Pooling, CMN, Centroid, Identification & Verification
│   └── separation.py             <- Instantaneous mixture, FastICA, xử lý Permutation & Sign, tương quan
└── results/
    ├── metrics_summary.csv       <- Bảng số liệu định lượng tự động ghi nhận từ 8 thí nghiệm
    ├── audio/                    <- Toàn bộ file âm thanh đầu ra:
    │   ├── noisy_5dB.wav         <- Câu thoại lẫn tiếng quạt gió tại SNR=5dB
    │   ├── enhanced_ss.wav       <- Sau khi khử nhiễu bằng Spectral Subtraction
    │   ├── enhanced_wiener.wav   <- Sau khi khử nhiễu bằng Wiener Filter
    │   ├── mix_mic1.wav          <- Tín hiệu hỗn hợp quan sát tại Micro 1
    │   ├── mix_mic2.wav          <- Tín hiệu hỗn hợp quan sát tại Micro 2
    │   ├── separated_1.wav       <- Nguồn tiếng nói thứ nhất sau khi tách bằng FastICA
    │   └── separated_2.wav       <- Nguồn tiếng nói thứ hai sau khi tách bằng FastICA
    └── figures/                  <- Toàn bộ biểu đồ khoa học chất lượng cao:
        ├── fig1_enhancement_spectrograms.png <- Phổ Spectrogram: Clean, Noisy, SS, Wiener
        ├── fig3_confusion_matrix.png         <- Ma trận nhầm lẫn Speaker Identification
        ├── fig4_far_frr_curve.png            <- Đường cong Trade-off FAR - FRR theo ngưỡng theta
        └── fig5_ica_correlation.png          <- Ma trận tương quan tuyệt đối ICA trước và sau khử bất định
```

---

## 2. Hướng dẫn Cài đặt & Tái lập Kết quả

### 2.1 Cài đặt thư viện phụ thuộc
Đảm bảo đã kích hoạt môi trường Python (khuyến nghị 3.10+):
```powershell
pip install -r requirements.txt
```

### 2.2 Sinh bộ dữ liệu âm thanh chuẩn 16 kHz
```powershell
python generate_dataset.py
```

### 2.3 Thực thi toàn bộ thí nghiệm & xuất kết quả
Chạy lệnh sau để tự động chạy Case Study "Trợ lý phòng họp mini", 8 thí nghiệm bắt buộc (TN1 – TN8), xuất các file WAV, biểu đồ PNG và file số liệu CSV:
```powershell
python run_lab5.py
```

### 2.4 Mở và tương tác với Jupyter Notebook
Khởi chạy Jupyter Notebook hoặc VS Code để xem `Lab05.ipynb` đã được chạy sẵn toàn bộ kết quả, biểu đồ và trình phát âm thanh:
```powershell
jupyter notebook Lab05.ipynb
```

---

## 3. Tóm tắt Kết quả Định lượng Nổi bật
- **Khử nhiễu STFT tại SNR=5dB:**
  - Spectral Subtraction: $\text{SNR}_{\text{out}} = 8.61\text{ dB} \implies \Delta\text{SNR} = \mathbf{+3.61\text{ dB}}$.
  - Wiener Filter: $\text{SNR}_{\text{out}} = 8.50\text{ dB} \implies \Delta\text{SNR} = \mathbf{+3.50\text{ dB}}$ (âm sắc tự nhiên, triệt tiêu musical noise).
- **Speaker Recognition:**
  - Identification: Accuracy đạt **$100.0\%$** trên tập Test độc lập.
  - Verification: Ngưỡng tối ưu $\theta^* = 0.9950$ tại điểm EER ($0.00\%$), đánh giá trên tập Test đạt $\text{FAR} = 0.00\%$, $\text{FRR} = 0.00\%$.
- **FastICA Separation:**
  - Phân tách hoàn hảo 2 nguồn nói chồng lấn qua ma trận trộn $A = \begin{bmatrix} 1.0 & 0.6 \\ 0.4 & 1.0 \end{bmatrix}$ ($\det(A) = 0.7600$).
  - Tự động khử hoán vị $[1, 0]$ và đảo dấu pha, đưa hệ số tương quan đạt **$1.0000$** tuyệt đối với cả 2 nguồn gốc.
