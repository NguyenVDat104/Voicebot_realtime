# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 3
## NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM (HIDDEN MARKOV MODEL)
**Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo – Trường Đại học Thủy lợi**  
**Sinh viên thực hiện:** Nguyễn Văn Đạt | **MSSV:** 215106xxxx

---

## 1. Giới Thiệu Tổng Quan
Dự án triển khai hoàn chỉnh hệ thống nhận dạng tiếng nói từ đơn (Isolated Word Recognition) cho 5 lệnh điều khiển tiếng Việt: `mở`, `đóng`, `lên`, `xuống`, `dừng` dựa trên mô hình **Gaussian Hidden Markov Model (HMM)** với:
- Vector đặc trưng 39 chiều: $13\text{ MFCC} + 13\ \Delta + 13\ \Delta^2$ kết hợp chuẩn hóa **Utterance-level CMVN**.
- Mô hình **Left-to-Right 5 trạng thái (Bakis Topology)** với khởi tạo **Linear Temporal Segmentation**.
- Thuật toán **Forward** tính toán Log-likelihood và **Viterbi** truy vết chuỗi trạng thái tối ưu (Trellis Alignment).
- Đối chuẩn thực nghiệm chi tiết với **DTW Baseline (Lab 2)**.
- Mở rộng **Pronunciation Lexicon** (`lexicon.txt`) và **Bigram Language Model** (`corpus.txt`) làm mịn Laplace Add-k ($k=0.5$) thực hiện giải mã kết hợp Bayes (MAP Re-ranking).
- Thực thi toàn bộ chuỗi **6 thí nghiệm bắt buộc (E1 $\rightarrow$ E6)** và xây dựng cơ chế lọc từ ngoài tập (**Reject Out-of-Vocabulary**).

---

## 2. Cấu Trúc Thư Mục `lab3/`
```text
Speech&Audio/lab3/
├── PLAN.md                     <- Kế hoạch tổng thể và lộ trình triển khai chi tiết
├── README.md                   <- Hướng dẫn cài đặt, cấu trúc thư mục & tái lập kết quả
├── requirements.txt            <- Danh sách các thư viện cần thiết
├── generate_dataset.py         <- Kịch bản sinh dữ liệu âm thanh chuẩn 16kHz qua gTTS
├── record_dataset.py           <- Công cụ thu âm trực tiếp qua Microphone
├── run_lab3.py                 <- Kịch bản thực thi toàn bộ pipeline A -> H và xuất kết quả
├── Lab03.ipynb                 <- Jupyter Notebook tương tác hoàn chỉnh
├── report_Lab03.md             <- Báo cáo học thuật chi tiết 4-6 trang, giải quyết 10 câu hỏi 6.1
├── make_notebook.py            <- Kịch bản tự động sinh Lab03.ipynb
├── lexicon.txt                 <- Từ điển phát âm chuẩn WORD PHONE_1...
├── corpus.txt                  <- Ngữ liệu câu lệnh huấn luyện Bigram LM
├── results.csv                 <- Bảng số liệu định lượng chi tiết các thí nghiệm E1 -> E6
├── dataset/                    <- Kho dữ liệu âm thanh chuẩn 16kHz mono PCM 16-bit
│   ├── train/                  <- 6 file/từ x 5 từ = 30 file (mo, dong, len, xuong, dung)
│   ├── test/                   <- 2 file/từ x 5 từ = 10 file
│   └── test_speaker2/          <- 10 file kiểm thử Mismatch người nói
├── models/                     <- 5 file mô hình HMM serialized (.pkl)
│   ├── hmm_mo.pkl
│   ├── hmm_dong.pkl
│   ├── hmm_len.pkl
│   ├── hmm_xuong.pkl
│   └── hmm_dung.pkl
└── figures/                    <- Đồ thị phân tích định dạng chất lượng cao
    ├── fig1_mfcc39_heatmaps.png          <- Heatmap so sánh đặc trưng 39-D của 'mở' và 'đóng'
    ├── fig2_viterbi_alignment_mo.png     <- Viterbi Trellis alignment cho lệnh 'mở'
    ├── fig3_viterbi_alignment_dong.png   <- Viterbi Trellis alignment cho lệnh 'đóng'
    ├── fig4_confusion_matrix_hmm.png     <- Ma trận nhầm lẫn Baseline HMM
    ├── fig5_dtw_vs_hmm_comparison.png    <- Biểu đồ so sánh DTW vs HMM theo số mẫu K
    ├── fig6_lm_reranking_analysis.png    <- Biểu đồ MAP Re-ranking theo trọng số lambda_LM
    └── fig7_score_distributions_reject.png <- Phân bố điểm số & Ngưỡng Reject Threshold
```

---

## 3. Hướng Dẫn Cài Đặt Môi Trường

1. **Kích hoạt môi trường Python (Python 3.10 trở lên):**
   ```bash
   pip install -r requirements.txt
   ```
   *Các thư viện chính:* `hmmlearn`, `librosa`, `soundfile`, `scikit-learn`, `matplotlib`, `gtts`, `joblib`.

2. **Sinh lại bộ dữ liệu mẫu (Tùy chọn):**
   ```bash
   python generate_dataset.py
   ```
   *(Hoặc thu âm bằng microphone qua `python record_dataset.py`).*

---

## 4. Hướng Dẫn Thực Thi và Tái Lập Kết Quả

### Cách 1: Chạy toàn bộ pipeline tự động qua Terminal
```bash
python run_lab3.py
```
Lệnh trên sẽ tự động:
1. Trích xuất đặc trưng 39-D ($13\text{ MFCC} + 13\ \Delta + 13\ \Delta^2$) và vẽ Heatmap `fig1_mfcc39_heatmaps.png`.
2. Huấn luyện 5 mô hình Left-to-Right Gaussian HMM, kiểm tra hội tụ và lưu mô hình vào `models/`.
3. Đánh giá tập test sạch, vẽ ma trận nhầm lẫn `fig4_confusion_matrix_hmm.png` và Viterbi Alignment `fig2_...`, `fig3_...`.
4. Chạy đối chuẩn DTW Baseline và xuất đồ thị so sánh `fig5_dtw_vs_hmm_comparison.png`.
5. Đọc `lexicon.txt`, tính Bigram LM từ `corpus.txt`, thực hiện MAP Re-ranking `fig6_...`.
6. Thực hiện đầy đủ 6 thí nghiệm bắt buộc (E1 $\rightarrow$ E6), xác định Reject Threshold `fig7_...` và xuất `results.csv`.

### Cách 2: Trải nghiệm tương tác trên Jupyter Notebook
Mở tệp [Lab03.ipynb](file:///c:/Speech&Audio/lab3/Lab03.ipynb) bằng Jupyter Notebook hoặc VS Code / IDE:
```bash
jupyter notebook Lab03.ipynb
```
Chạy tuần tự các ô lệnh từ trên xuống dưới để quan sát trực tiếp từng bước phân tích và biểu đồ trực quan.

---

## 5. Tóm Tắt Kết Quả Định Lượng Nổi Bật

- **Độ chính xác nhận dạng HMM Baseline (Test sạch):** **$100.0\%$ (10/10 file)**.
- **Biên độ tự tin phân loại:** Khoảng cách Log-likelihood giữa Top-1 và Top-2 vượt trên $+1100$ điểm.
- **Tốc độ suy diễn:** Gaussian HMM nhanh hơn DTW gấp gần **7 lần** ($1.8\text{ ms/mẫu}$ so với $12.4\text{ ms/mẫu}$).
- **Khảo sát Mismatch (E5):** Duy trì $100\%$ độ chính xác trên người nói mới (Speaker 2); giảm xuống $40\%$ khi có nhiễu trắng mạnh SNR 15dB.
- **Ngưỡng lọc từ ngoài tập (Reject Threshold):** $\theta_{\text{score}} = -46.20$ điểm/frame và $\theta_{\text{margin}} = 0.15$.

Chi tiết lý thuyết và giải quyết 10 câu hỏi báo cáo được trình bày đầy đủ tại: [report_Lab03.md](file:///c:/Speech&Audio/lab3/report_Lab03.md).
