# KẾ HOẠCH TRIỂN KHAI THỰC HÀNH LAB 4
## TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH - TTS)
### Từ Kiến trúc TTS, Ghép nối & Unit Selection đến Tổng hợp Tham số HMM và TTS Học sâu
**Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói  
**Bộ môn:** Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin – Trường Đại học Thủy lợi  
**Sinh viên thực hiện:** Nguyễn Văn Đạt  
**Mục tiêu tài liệu:** Bản kế hoạch tổng thể (Master Plan) chuẩn hóa luồng xử lý, đặc tả toán học, phân bổ thí nghiệm và tiêu chí bàn giao nhằm đạt điểm tối đa (10/10) theo rubric Lab 4.

---

## I. TỔNG QUAN VÀ BỐI CẢNH KỸ THUẬT

### 1. Vị trí của Lab 4 trong chuỗi học phần CSE457
* **Lab 1 (Tín hiệu & Biến đổi):** Nền tảng xử lý tín hiệu rời rạc: Waveform, Sampling rate, FFT/STFT, Windowing, Spectrogram và Overlap-Add (OLA).
* **Lab 2 & Lab 3 (Phân tích & Nhận dạng - ASR):** Trích xuất đặc trưng âm học (F0, Energy, MFCC, $\Delta, \Delta\Delta$), mô hình hóa thời gian với DTW và mô hình thống kê Gaussian HMM để chuyển từ tiếng nói thành văn bản ($Waveform \rightarrow Text$).
* **Lab 4 (Tổng hợp tiếng nói - TTS):** Bài toán đảo ngược toàn diện ($Text \rightarrow Waveform$). Sinh viên liên kết toàn bộ kiến thức xử lý tín hiệu, trắc lượng âm học (prosody), mô hình nguồn-lọc (source-filter) và mô hình xác suất/học sâu để sinh waveform có ngữ điệu và tính tự nhiên.

### 2. Ba trường phái tổng hợp tiếng nói được đối sánh
1. **Concatenative TTS (Tổng hợp ghép nối & Unit Selection):**
   - *Nguyên lý:* Cắt các mẫu thu âm tự nhiên (word, diphone, phone) thành kho lưu trữ (inventory). Khi tổng hợp một câu mới, thuật toán tìm kiếm đường đi tối ưu (Viterbi / Exhaustive Search) trên lưới candidate để cực tiểu hóa tổng Target Cost và Join Cost, sau đó nối lại và làm mịn biên (Cross-fade / OLA).
   - *Ưu điểm:* Chất lượng âm sắc tự nhiên cao ở từng đoạn đơn lẻ.
   - *Nhược điểm:* Đòi hỏi cơ sở dữ liệu rất lớn; dễ xuất hiện click, discontinuity biên và không linh hoạt điều chỉnh ngữ điệu (prosody).
2. **HMM-based Parametric TTS (Tổng hợp tham số thống kê):**
   - *Nguyên lý:* HMM chạy ở chế độ **sinh (generation mode)**: Text $\rightarrow$ Context phone $\rightarrow$ State sequence. Mỗi trạng thái phát ra phân bố tham số âm học (F0, spectral/formant, duration). Nghiệm hợp lý cực đại kết hợp ràng buộc động học ($\Delta, \Delta\Delta$) tạo ra quỹ đạo tham số (parameter trajectory) trơn mượt theo thời gian, sau đó đưa qua bộ tổng hợp (Vocoder).
   - *Ưu điểm:* Mô hình cực nhẹ, dễ dàng điều khiển cao độ (pitch), tốc độ (duration), cảm xúc.
   - *Nhược điểm:* Giọng nghe có tính "buzzy", hơi nghẹt hoặc có cảm giác robot hóa do quá trình làm trơn phổ trung bình.
3. **Neural TTS (TTS học sâu hiện đại):**
   - *Nguyên lý:* Mô hình học sâu chia làm 2 giai đoạn: Acoustic Model (sinh mel-spectrogram từ text, ví dụ FastSpeech2 / Tacotron2) $\rightarrow$ Neural Vocoder (tổng hợp waveform từ mel-spectrogram, ví dụ HiFi-GAN / WaveGlow / Vocos).
   - *Ưu điểm:* Độ tự nhiên vượt trội, ngữ điệu chân thực như người thật, mở rộng đa người nói tốt.
   - *Nhược điểm:* Đòi hỏi tài nguyên tính toán lớn (GPU/CPU), chi phí huấn luyện cao và có độ trễ (latency).

---

## II. ĐẶC TẢ TOÁN HỌC VÀ CÔNG THỨC CỐT LÕI

### 1. Chuỗi biến đổi Text Analysis
$$T \xrightarrow{\text{normalize}} W = (w_1, w_2, \dots, w_K) \xrightarrow{\text{G2P/lexicon}} P = (p_1, p_2, \dots, p_L)$$
- $T$: Văn bản thô ban đầu (ví dụ: `"Mời khách hàng số 23 đến quầy 5."`).
- $W$: Chuỗi token miền hẹp đã chuẩn hóa chữ thường, loại bỏ dấu ngắt câu thừa và phân rã số đếm từng ký số: `["mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"]`.

### 2. Mô hình Nguồn – Lọc (Source–Filter Model)
$$X(z) = E(z) \cdot H(z) = E(z) \cdot \frac{B(z)}{A(z)} = E(z) \cdot \frac{b_0 + \sum_{k=1}^M b_k z^{-k}}{1 - \sum_{k=1}^P a_k z^{-k}}$$
- $E(z)$: Nguồn kích thích (Excitation). Với âm hữu thanh (voiced), $e[n]$ là chuỗi xung tuần hoàn theo chu kỳ pitch $N_p = F_s / F_0$; với âm vô thanh (unvoiced), $e[n]$ là nhiễu trắng ngẫu nhiên.
- $H(z)$: Bộ lọc thanh đạo (Vocal tract filter) quyết định các tần số cộng hưởng (Formant: $F_1, F_2, F_3$) và bao phổ (Spectral Envelope).
- **Ý nghĩa quan trọng:** Thay đổi $F_0$ (pitch) chỉ tác động vào tần số nguồn $E(z)$, tuyệt đối **không** được làm thay đổi vị trí các cực lọc formant của $H(z)$ (tránh hiệu ứng méo tiếng "chipmunk").

### 3. Tối ưu hóa Unit Selection
Chuỗi unit tối ưu $\theta^* = (\theta_1^*, \dots, \theta_J^*)$ được tìm qua hàm mục tiêu cực tiểu tổng chi phí:
$$\theta^* = \arg\min_{\{\theta_1, \dots, \theta_J\}} \left[ \sum_{j=1}^J C_{\text{target}}(\theta_j, t_j) + \lambda \sum_{j=1}^{J-1} C_{\text{join}}(\theta_j, \theta_{j+1}) \right]$$
- **Target Cost $C_{\text{target}}(\theta_j, t_j)$:** Đo mức độ sai lệch giữa ứng viên $\theta_j$ với yêu cầu mục tiêu $t_j$:
  $$C_{\text{target}}(\theta_j, t_j) = w_d \frac{|d(\theta_j) - d_{\text{target}}|}{\sigma_d} + w_{\text{rms}} \frac{|\text{RMS}(\theta_j) - \text{RMS}_{\text{target}}|}{\sigma_{\text{rms}}}$$
- **Join Cost $C_{\text{join}}(\theta_j, \theta_{j+1})$:** Đo độ gián đoạn phổ tại biên ghép (spectral discontinuity) lấy trên cửa sổ $N$ mẫu (20–30 ms) ở cuối unit $j$ và đầu unit $j+1$:
  $$D_{\text{spec}} = \frac{1}{K} \sum_{k=1}^K \left[ L_{\text{left}}[k] - L_{\text{right}}[k] \right]^2$$
  Trong đó:
  - $L_{\text{left}}[k] = 20 \log_{10} \max\left( \left| \text{RFFT}(a[-N:] \cdot w) \right|, 10^{-8} \right)$
  - $L_{\text{right}}[k] = 20 \log_{10} \max\left( \left| \text{RFFT}(b[:N] \cdot w) \right|, 10^{-8} \right)$
  - Có thể cộng thêm độ lệch RMS và F0: $C_{\text{join}} = D_{\text{spec}} + \beta_1 |\Delta \text{RMS}| + \beta_2 |\Delta F_0|$.

### 4. Làm mịn biên (Boundary Smoothing)
- **Kỹ thuật Cross-fade:** Trộn tuyến tính biên chuyển tiếp trên đoạn $N$ mẫu ($N = F_s \cdot \tau_{\text{xfade}} / 1000$):
  $$y[n] = (1 - \alpha[n]) \cdot x_1[n] + \alpha[n] \cdot x_2[n], \quad 0 \le n < N, \quad \alpha[n] = \frac{n}{N-1}$$
- **Chuyển đổi cao độ (Pitch Shift):**
  $$r = 2^{\Delta s / 12}, \quad F_{0,\text{new}} = r \cdot F_{0,\text{old}}$$
  ($\Delta s$: số bán âm semitones; $\Delta s = \pm 2 \Rightarrow r \approx 2^{\pm 2/12} \approx 1.122 / 0.891$).

### 5. Sinh chuỗi tham số trong HMM (Generation Mode)
$$\hat{O}^* = \arg\max_{O} P(O \mid \lambda, Q) \quad \text{với } O_t = [c_t, \Delta c_t]^T$$
Ràng buộc động học $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$ buộc nghiệm hợp lý cực đại tạo thành quỹ đạo trơn liên tục, loại bỏ hiện tượng nhảy bậc thang giữa các trạng thái rời rạc.

---

## III. THIẾT KẾ DỮ LIỆU VÀ CẤU TRÚC THƯ MỤC

### 1. Case Study: Miền hẹp “Thông báo xếp hàng”
* **Miền từ vựng mục tiêu (16 từ vựng):**
  - Cụm điều khiển: `mời`, `khách`, `hàng`, `số`, `đến`, `quầy` (6 từ).
  - Cụm chữ số: `không`, `một`, `hai`, `ba`, `bốn`, `năm`, `sáu`, `bảy`, `tám`, `chín` (10 chữ số).
* **Quy cách thu âm/sinh mẫu của Inventory:**
  - Định dạng: WAV mono, $F_s = 16,000\text{ Hz}$, PCM 16-bit.
  - Số lượng: Tối thiểu 2–3 bản ghi (candidate) cho mỗi từ $\Rightarrow$ Tổng cộng từ 32 đến 48 file WAV.
  - Xử lý tiền kỳ: Cắt bỏ silence đầu/cuối sạch sẽ (để lại 20–30 ms padding), chuẩn hóa RMS về mức mục tiêu 0.08.
* **Các câu test tiêu chuẩn:**
  - **Câu 1 (Mẫu chính):** `"Mời khách hàng số 23 đến quầy 5."`
  - **Câu 2 (Kiểm tra số đa dạng):** `"Mời khách hàng số 104 đến quầy 2."`
  - **Câu 3 (Kiểm tra dãy số dài):** `"Mời khách hàng số 876 đến quầy 9."`

### 2. Sơ đồ cấu trúc thư mục mục tiêu `lab4/`
```text
Speech&Audio/lab4/
├── Lab 4.pdf                    <- Đề bài gốc Lab 4
├── PLAN.md                     <- Kế hoạch tổng thể chi tiết này
├── README.md                   <- Hướng dẫn cài đặt môi trường và lệnh thực thi
├── requirements.txt            <- Thư viện cần thiết (numpy, scipy, matplotlib, soundfile, librosa)
├── generate_inventory.py       <- Script tự động tạo/chuẩn hóa kho âm thanh 16kHz
├── hmm_demo.py                 <- Backend mô phỏng HMM trajectory generation
├── neural_tts_backend.py       <- Wrapper offline Neural TTS (chạy mô hình định sẵn, đo latency)
├── run_lab4.py                 <- Mã nguồn Python thực thi tự động từ A -> G, xuất audio và biểu đồ
├── Lab04.ipynb                 <- Jupyter Notebook tương tác trực quan 3 tiết thực hành
├── report_Lab04.md             <- Báo cáo học thuật 5-8 trang, trả lời 10 câu hỏi lý thuyết
├── ratings.csv                 <- Bảng kết quả đánh giá MOS / AB test từ người nghe
├── lab4_data/
│   ├── units/                  <- Kho WAV đơn vị từ vựng (2-3 candidates/unit)
│   │   ├── moi_01.wav, moi_02.wav
│   │   ├── khach_01.wav, khach_02.wav
│   │   ├── hang_01.wav, hang_02.wav
│   │   ├── so_01.wav, so_02.wav
│   │   ├── khong_01.wav ... chin_02.wav
│   │   ├── den_01.wav, den_02.wav
│   │   └── quay_01.wav, quay_02.wav
│   └── metadata.csv            <- Bảng thông tin: unit_label, candidate_id, duration, rms, file_path
├── audio_output/               <- Folder xuất các file âm thanh sản phẩm
│   ├── raw_concat.wav          <- Kết quả ghép nối baseline (xfade = 0ms)
│   ├── best_concat.wav         <- Kết quả sau unit selection + cross-fade tối ưu (15ms)
│   ├── xfade_05ms.wav          <- Kiểm thử cross-fade 5ms
│   ├── xfade_30ms.wav          <- Kiểm thử cross-fade 30ms
│   ├── prosody_speed_08x.wav   <- Time-stretch tốc độ 0.8x
│   ├── prosody_speed_12x.wav   <- Time-stretch tốc độ 1.2x
│   ├── prosody_pitch_down2.wav <- Pitch shift -2 semitones
│   ├── prosody_pitch_up2.wav   <- Pitch shift +2 semitones
│   ├── hmm_tts.wav             <- Kết quả từ HMM-based synthesis
│   └── neural_tts.wav          <- Kết quả từ Pretrained Neural TTS
└── figures/                    <- Tối thiểu 5 hình ảnh chất lượng cao phục vụ báo cáo
    ├── fig1_inventory_stats.png        <- Phân phối thời lượng (duration) và năng lượng (RMS)
    ├── fig2_spectrogram_boundaries.png <- So sánh waveform & spectrogram trước/sau cross-fade (có vạch biên)
    ├── fig3_unit_selection_matrix.png  <- Ma trận join cost và lưới Dynamic Programming Trellis
    ├── fig4_prosody_modifications.png  <- Phân tích F0 contour, Pitch Shift và Time-Stretch
    ├── fig5_hmm_trajectories.png       <- Quỹ đạo tham số HMM (F0 & Formant/Cepstrum tracks)
    └── fig6_comparative_spectrogram.png<- So sánh đầu ra 3 phương pháp (Concat vs HMM vs Neural)
```

---

## IV. PHÂN KỲ TRIỂN KHAI CHI TIẾT (PHASE 1 ĐẾN PHASE 5)

### Phase 1: Môi trường, Chuẩn hóa Dữ liệu & Text Normalization (Yêu cầu 4.1 - A)
1. **Khởi tạo môi trường Python:**
   - Cài đặt các thư viện: `numpy`, `scipy`, `matplotlib`, `soundfile`, `librosa`.
2. **Xây dựng module Chuẩn hóa Văn bản (`normalize_text`):**
   - Chuyển văn bản về lowercase, lọc sạch các dấu câu `[.,!?;:]`.
   - Xây dựng từ điển `DIGIT_MAP` ánh xạ từ ký tự số sang chữ (`'0' -> 'không'`, `'1' -> 'một'`, ..., `'9' -> 'chín'`).
   - Phân tích token số và chuyển đổi ngữ cảnh miền hẹp (đọc tách từng chữ số cho mã xếp hàng).
3. **Thu thập / Khởi tạo Kho Inventory:**
   - Đảm bảo đủ 16 từ vựng $\times$ 2 candidate/từ = 32 file WAV sạch.
   - Chuẩn hóa: $F_s = 16\text{ kHz}$, mono, kiểu float32.
   - Cân bằng năng lượng bằng hàm `match_rms(x, target=0.08)` và tránh clipping bằng `np.clip(y, -0.99, 0.99)`.
   - Xuất file `lab4_data/metadata.csv` chứa thông số của toàn bộ kho âm thanh.
4. **Vẽ biểu đồ phân tích dữ liệu:**
   - Xuất `fig1_inventory_stats.png`: Histogram phân bố duration và histogram RMS để phát hiện điểm bất thường (outliers).

### Phase 2: Ghép nối Baseline & Phân tích Gián đoạn Biên (Yêu cầu 4.2 - B)
1. **Triển khai ghép nối đơn giản (Direct Concatenation):**
   - Cố định chọn candidate `_01.wav` cho mỗi từ trong chuỗi token.
   - Ghép trực tiếp bằng phép nối mảng numpy thông thường (không làm mịn): $y = [x_1, x_2, \dots, x_K]$.
   - Xuất file `audio_output/raw_concat.wav`.
2. **Phân tích vị trí biên nối (Boundary Discontinuity):**
   - Lưu trữ mảng tọa độ thời gian các điểm nối: $t_{\text{boundary}} = [T_1, T_1+T_2, \dots]$.
   - Vẽ waveform và STFT spectrogram tương ứng, kẻ đường gióng thẳng đứng màu đỏ đánh dấu chính xác vị trí biên.
3. **Thính chẩn và ghi nhận lỗi thô:**
   - Đánh giá hiện tượng click/pop do lệch pha hoặc lệch giá trị biên độ tức thời tại điểm nối.
   - Ghi nhận hiện tượng pitch jump (bước nhảy cao độ đột ngột) và biến thiên đột ngột của bao phổ giữa hai từ liền kề.

### Phase 3: Unit Selection & Làm mịn Biên (Yêu cầu 4.3 - C)
1. **Định nghĩa các hàm chi phí (Cost Functions):**
   - Xây dựng `join_cost(a, b)`: Trích xuất $N$ mẫu ($25\text{ ms}$) ở cuối unit $a$ và đầu unit $b$, nhân cửa sổ Hanning, tính RFFT log-magnitude spectrum $L_a, L_b$ và trả về sai số trung bình bình phương (MSE).
   - Xây dựng `target_cost(cand, target)` dựa trên chênh lệch duration và RMS so với giá trị trung bình mong muốn.
2. **Thuật toán Chọn Unit Tối ưu (Dynamic Programming / Viterbi Search):**
   - Xây dựng bảng quy hoạch động trên lưới ứng viên: Tại mỗi vị trí từ $j$, tính toán tổng chi phí tích lũy từ mọi candidate ở bước $j-1$ và lưu vết `backpointer`.
   - Truy vết ngược để lấy ra chuỗi ứng viên có tổng chi phí toàn cục nhỏ nhất: $\theta^*$.
   - Lập bảng so sánh chi phí tổng của: (1) Chuỗi chọn cố định (Baseline) vs (2) Chuỗi tối ưu qua DP.
3. **Thí nghiệm kỹ thuật Cross-fade:**
   - Triển khai hàm `crossfade(a, b, n)` với ramp tuyến tính $\alpha[n] = n / (n - 1)$.
   - Khảo sát 4 mức thời gian overlap: $0\text{ ms}$, $5\text{ ms}$, $15\text{ ms}$, $30\text{ ms}$.
   - Xuất các file âm thanh tương ứng và vẽ đồ thị so sánh `fig2_spectrogram_boundaries.png`.
   - Chọn ra cấu hình tối ưu nhất để lưu file `audio_output/best_concat.wav`.

### Phase 4: Trắc lượng Âm học (Prosody), HMM Generation & Neural TTS (Yêu cầu 4.4 - D, 4.5 - E, 4.6 - F)
1. **Khảo sát Prosody và Source-Filter (Mục D):**
   - Chọn một nguyên âm hữu thanh (ví dụ: đoạn giữa của từ `"số"` hoặc `"hai"`).
   - Trích xuất F0 qua thuật toán tự tương quan (Autocorrelation) và kiểm chứng công thức $F_0 \approx F_s / N_p$.
   - Thực hiện Time-stretch biến đổi tốc độ ($0.8\times$ và $1.2\times$) bằng WSOLA/Phase Vocoder.
   - Thực hiện Pitch Shift dịch cao độ $\pm 2$ bán âm ($r = 2^{\pm 2/12}$).
   - Phân tích hiện tượng: Vì sao thay đổi cao độ mà không giữ nguyên bao phổ formant sẽ làm biến tính giọng người nói.
2. **Mô phỏng HMM-based Synthesis (Mục E):**
   - Xây dựng module `hmm_demo.py` mô phỏng cơ chế phát sinh tham số âm học của HMM: Từ chuỗi từ sinh chuỗi trạng thái (State Sequence) và phát ra quỹ đạo $F_0(t)$, $c_1(t)$ (formant/cepstral trajectory).
   - Thể hiện trực quan tác dụng của hệ số động học $\Delta$: Giúp đường cong tham số trơn tru liên tục thay vì bậc thang thô.
   - Thực hiện tổng hợp tham số ra file `audio_output/hmm_tts.wav`.
3. **Inference Neural TTS Offline (Mục F):**
   - Đóng gói wrapper `neural_tts_backend.py` chạy hoàn toàn offline (sử dụng Edge-TTS pre-cached / Pyttsx3 / Tacotron2-Vocoder offline checkpoint).
   - Chạy inference cho 3 câu thử nghiệm tiêu chuẩn.
   - Đo đạc chính xác thời gian xử lý (Latency) từ lúc nạp văn bản đến khi xuất mảng waveform.
   - Trích xuất mel-spectrogram và lưu file âm thanh `audio_output/neural_tts.wav`.

### Phase 5: Đánh giá Toàn diện, So sánh Đối chuẩn & Báo cáo (Yêu cầu 4.7 - G & Thí nghiệm 1-5)
1. **Tiến hành 5 Thí nghiệm Bắt buộc (TN 1 -> TN 5):**
   - *TN1 (Cross-fade):* Đánh giá định tính và định lượng 4 mức $0, 5, 15, 30\text{ ms}$.
   - *TN2 (Unit Selection):* So sánh 1 candidate vs 2–3 candidates trên cùng một câu.
   - *TN3 (Tốc độ):* Đánh giá độ tự nhiên và độ rõ khi thay đổi tốc độ $0.8\times, 1.0\times, 1.2\times$.
   - *TN4 (Cao độ):* Thính chẩn hiện tượng méo tiếng (artifact/buzziness) khi dịch pitch $\pm 2$ semitones.
   - *TN5 (So sánh 3 Backend):* Chấm điểm 5 tiêu chí: Intelligibility, Naturalness, Continuity, Latency, Bộ nhớ.
2. **Tổ chức Đánh giá Chủ quan (Listening Test / MOS):**
   - Thu thập điểm đánh giá từ 5–10 người nghe theo thang MOS (1: Rất tệ -> 5: Rất tự nhiên).
   - Tính toán độ rõ từ (Intelligibility % = $100 \times N_{\text{correct}} / N_{\text{total}}$).
   - Xuất dữ liệu vào file `ratings.csv`.
3. **Hoàn thiện Báo cáo học thuật (`report_Lab04.md`):**
   - Cung cấp lời giải chi tiết và thấu đáo cho toàn bộ **10 câu hỏi báo cáo**.
   - Chèn đầy đủ tối thiểu 5 hình đồ thị chất lượng cao và bảng số liệu so sánh.

---

## V. MA TRẬN 5 THÍ NGHIỆM BẮT BUỘC

| STT | Tên Thí nghiệm | Biến thay đổi | Dải giá trị thử nghiệm | Chỉ số / Phương pháp quan sát | Câu hỏi nghiên cứu cần trả lời |
|:---:|:---|:---|:---|:---|:---|
| **TN1** | Tối ưu hóa Cross-fade | Độ dài cửa sổ chồng lấn $\tau_{\text{xfade}}$ | $0\text{ ms}$, $5\text{ ms}$, $15\text{ ms}$, $30\text{ ms}$ | Waveform biên, Spectrogram gián đoạn phổ, Nghe click | Mức cross-fade nào triệt tiêu click tốt nhất mà không làm nhòe/mờ phụ âm chuyển tiếp? |
| **TN2** | Hiệu quả Unit Selection | Số lượng Candidate/Unit | $1$ candidate (cố định) vs $2-3$ candidates (Dynamic Programming) | Tổng Join Cost, Target Cost, Độ trơn mượt khi nghe | Unit selection dựa trên DP có thực sự giảm thiểu join cost và nâng cao độ tự nhiên so với ghép ngẫu nhiên? |
| **TN3** | Điều chỉnh Duration / Tốc độ | Hệ số co giãn thời gian $\alpha_t$ | $0.8\times$ (nhanh), $1.0\times$ (gốc), $1.2\times$ (chậm) | Thời lượng câu (s), Điểm Intelligibility (%) | Tốc độ phát âm ảnh hưởng thế nào đến độ dễ nghe và độ tự nhiên trong bài toán thông báo hàng đợi? |
| **TN4** | Biến đổi Cao độ (Pitch Shift) | Độ dịch bán âm $\Delta s$ | $-2$ semitones, $0$, $+2$ semitones | $F_0$ contour trích xuất, Thính chẩn artifact phổ | Pitch shift có gây ra hiện tượng rè tiếng (buzziness) hoặc làm giọng nói biến dạng thành "người máy" không? |
| **TN5** | Đánh giá Đối chuẩn 3 Backend | Kiến trúc tổng hợp | **Concatenative** vs **HMM Parametric** vs **Neural TTS** | Bảng 5 tiêu chí (Rõ, Tự nhiên, Mịn, Trễ, Kích thước) | Ưu và nhược điểm cố hữu của từng phương pháp trong bài toán thực tế là gì? |

---

## VI. BỘ 10 CÂU HỎI BÁO CÁO VÀ ĐỊNH HƯỚNG TRẢ LỜI

1. **Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?**
   - *Định hướng:* Do sự thiếu nhất quán về ngữ điệu toàn câu (F0 contour nhảy vọt), chênh lệch năng lượng (RMS mismatch), và sự gián đoạn cấu âm (coarticulation mismatch) tại biên ghép khiến não bộ nhận diện âm thanh bị chắp vá gượng gạo.
2. **Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?**
   - *Định hướng:* Target cost đo độ phù hợp giữa ứng viên với yêu cầu ngữ cảnh mục tiêu (duration, pitch, stress); Join cost đo độ êm mượt phổ và pha tại điểm tiếp giáp 2 unit. Nếu chỉ tối ưu join cost, hệ thống có thể chọn chuỗi unit nối rất êm nhưng sai lệch hoàn toàn ngữ điệu hoặc chọn sai biến thể phát âm mong muốn.
3. **Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?**
   - *Định hướng:* Phụ âm (đặc biệt là âm tắc như /t/, /k/, /p/ hoặc âm xát) có thời lượng chuyển tiếp rất ngắn (10-30 ms) và đặc trưng năng lượng bùng nổ (burst). Cross-fade quá dài sẽ làm triệt tiêu pha, nhòe các đỉnh bùng nổ phổ (smearing) khiến phụ âm nghe như bị nuốt tiếng hoặc nghẹt.
4. **Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.**
   - *Định hướng:* Trong mô hình $X(z) = E(z)H(z)$, $F_0$ là tần số kích thích của nguồn dây thanh $E(z)$, còn formant là các tần số cộng hưởng của thanh đạo $H(z)$ phụ thuộc vào hình dạng khoang miệng/lưỡi. Thay $F_0$ chỉ thay đổi khoảng cách giữa các vạch điều hòa phổ chứ không làm dịch chuyển bao phổ hình học của các cực lọc formant.
5. **PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?**
   - *Định hướng:* TD-PSOLA xác định các pitch mark (epoch), nhân cửa sổ Hanning quanh mỗi epoch rồi dời khoảng cách giữa các epoch khi overlap-add. Với âm xát hữu thanh (voiced fricative), tín hiệu vừa có thành phần tuần hoàn vừa có thành phần nhiễu hỗn loạn; khi lặp lại nhân tạo các đoạn nhiễu theo chu kỳ tuần hoàn, thành phần nhiễu bị chu kỳ hóa cưỡng bức, gây ra tiếng rè "buzziness" cơ học.
6. **Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?**
   - *Định hướng:* Nếu không có delta, nghiệm cực đại hợp lý chỉ là các giá trị trung bình $\mu_i$ không đổi trong suốt thời gian tồn tại của trạng thái (tạo thành dạng sóng bậc thang). Vector gia tốc động $\Delta c_t$ đóng vai trò ràng buộc đạo hàm bậc nhất theo thời gian, ép nghiệm ML phải là một đường cong biến thiên liên tục mượt mà đi qua các phân bố.
7. **Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?**
   - *Định hướng:* Vì không thể tùy biến linh hoạt đặc tính âm học của waveform gốc sau khi thu. Để đạt độ tự nhiên cao cho mọi ngữ cảnh phát âm, kho dữ liệu phải chứa đầy đủ mọi biến thể âm vị kết hợp mọi ngữ điệu, trọng âm, và hiện tượng đồng hóa cấu âm (coarticulation).
8. **Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.**
   - *Định hướng:* (1) Concatenative dựa trên việc tìm kiếm và ghép các mẩu audio có sẵn (không sinh mới); Neural TTS sinh mới hoàn toàn biểu diễn âm học (mel-spectrogram) và waveform thông qua mạng nơ-ron sâu. (2) Concatenative bị ràng buộc bởi kích thước kho thu âm và khó chỉnh sửa giọng; Neural TTS có khả năng tổng quát hóa, nội suy ngữ điệu mượt mà và chuyển đổi giọng nói (voice cloning/adaptation) linh hoạt.
9. **Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?**
   - *Định hướng:* Mô hình tải ngẫu nhiên có sự khác biệt về tập dữ liệu huấn luyện, giọng đọc, sampling rate, chất lượng tiền xử lý và kiến trúc vocoder. So sánh không kiểm soát biến số sẽ gây thiếu khách quan, vi phạm điều kiện so sánh công bằng và làm lu mờ bản chất kỹ thuật cần đánh giá trong lab.
10. **Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.**
    - *Định hướng:* Có thể dùng trong các ứng dụng giải trí nền, âm thanh tạo bầu không khí (ambient voice, background crowd noise trong game, lồng tiếng nhân vật phụ hư cấu nói ngôn ngữ giả tưởng) hoặc các hệ thống phát sinh tiếng ru/thiền định; nơi người nghe chỉ cần cảm nhận giọng nói tự nhiên, êm dịu, ấm áp mà không quan tâm đến nội dung ngữ nghĩa chính xác của từng từ.

---

## VII. MA TRẬN ĐỐI CHIẾU RUBRIC CHẤM ĐIỂM (10 ĐIỂM)

| Hạng mục chấm | Điểm | Tiêu chuẩn đạt điểm tối đa | Minh chứng trong sản phẩm nộp |
|:---|:---:|:---|:---|
| **1. Chuẩn bị dữ liệu & normalization** | **1.0** | Kho từ vựng đầy đủ 16 từ vựng $\times$ 2-3 candidate; audio sạch cắt silence chuẩn; hàm `normalize_text` xử lý hoàn hảo số và văn bản; file `metadata.csv` rõ ràng; biểu đồ duration/RMS histogram đẹp mắt. | `lab4_data/metadata.csv`<br>`fig1_inventory_stats.png`<br>Hàm `normalize_queue_text()` |
| **2. Concatenative baseline** | **1.5** | Ghép thành công câu chuẩn `"Mời khách hàng số 23 đến quầy 5."` nguyên vẹn; xuất file `raw_concat.wav`; vẽ spectrogram có vạch kẻ biên màu đỏ; phân tích đầy đủ hiện tượng click và pitch jump. | `audio_output/raw_concat.wav`<br>`fig2_spectrogram_boundaries.png`<br>Mục phân tích hiện tượng biên |
| **3. Unit selection & smoothing** | **2.0** | Thiết lập hàm Join cost (spectral distance) và Target cost chính xác; thuật toán Dynamic Programming (Viterbi) chọn chuỗi tối ưu; thí nghiệm 4 mức cross-fade ($0, 5, 15, 30\text{ ms}$); xuất `best_concat.wav` êm mượt. | `audio_output/best_concat.wav`<br>`fig3_unit_selection_matrix.png`<br>Bảng chi phí so sánh DP vs Random |
| **4. Prosody / source–filter** | **1.0** | Đo $F_0$ thực nghiệm khớp công thức $F_0 \approx F_s / N_p$; thực hiện time-stretch và pitch shift $\pm 2$ bán âm; giải thích chính xác nguyên lý bảo toàn formant trong mô hình source-filter. | `audio_output/prosody_*.wav`<br>`fig4_prosody_modifications.png`<br>Phần lý giải khoa học |
| **5. HMM-based synthesis** | **1.5** | Trích xuất và vẽ trực quan quỹ đạo tham số $F_0$ contour và Formant track; giải thích rõ ràng vai trò của $\Delta$-feature trong việc làm trơn quỹ đạo; xuất file demo `hmm_tts.wav`. | `audio_output/hmm_tts.wav`<br>`fig5_hmm_trajectories.png`<br>Mục so sánh waveform vs trajectory |
| **6. Neural TTS inference** | **1.0** | Chạy mô hình offline ổn định cho cả 3 câu thử nghiệm; lưu file `neural_tts.wav` và mel-spectrogram tương ứng; đo đạc và báo cáo latency chính xác (ms). | `audio_output/neural_tts.wav`<br>`fig6_comparative_spectrogram.png`<br>Bảng đo độ trễ trên máy thực tế |
| **7. Đánh giá & Báo cáo** | **2.0** | Báo cáo khoa học 5-8 trang trình bày chỉn chu; trả lời toàn diện 10 câu hỏi lý thuyết; bảng kết quả nghe kiểm thử MOS/AB với file `ratings.csv`; phân tích sâu sắc các bảng biểu. | `report_Lab04.md`<br>`ratings.csv`<br>`Lab04.ipynb` chạy liền mạch từ đầu tới cuối |
| **TỔNG CỘNG** | **10.0** | **Đạt tiêu chuẩn xuất sắc toàn diện mọi tiêu chí đề ra của môn học.** | |

---

## VIII. KẾ HOẠCH HÀNH ĐỘNG VÀ CÁC BƯỚC TIẾP THEO

- [ ] **Bước 1:** Khởi tạo cấu trúc thư mục `lab4/` và các thư mục con: `lab4_data/units/`, `audio_output/`, `figures/`.
- [ ] **Bước 2:** Xây dựng script `generate_inventory.py` để tạo kho dữ liệu 16 từ vựng mẫu (chuẩn 16kHz, mono, RMS normalized) cùng file `metadata.csv`.
- [ ] **Bước 3:** Hiện thực hóa các hàm chuẩn hóa chuỗi `normalize_queue_text()`, các hàm trích xuất chi phí `join_cost()` và giải thuật Dynamic Programming chọn candidate.
- [ ] **Bước 4:** Xây dựng module xử lý Prosody (Autocorrelation F0, Time-stretch, Pitch-shift) và module mô phỏng HMM Trajectory Generation (`hmm_demo.py`).
- [ ] **Bước 5:** Thiết lập module Neural TTS Pretrained chạy offline (`neural_tts_backend.py`) và công cụ đo độ trễ (latency profiler).
- [ ] **Bước 6:** Đóng gói toàn bộ luồng vào `run_lab4.py` và Notebook `Lab04.ipynb` với đầy đủ code, markdown giải thích và biểu đồ trực quan.
- [ ] **Bước 7:** Tiến hành thu thập kết quả nghe MOS, xuất `ratings.csv` và hoàn thiện Báo cáo chi tiết `report_Lab04.md`.
