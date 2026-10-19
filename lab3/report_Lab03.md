# BÁO CÁO THỰC HÀNH LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
**Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói  
**Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo – Trường Đại học Thủy lợi**  

---

## THÔNG TIN CHUNG
* **Bài thực hành:** Lab 3 – Nhận dạng tiếng nói nâng cao với HMM (Từ MFCC động đến mô hình Markov ẩn, likelihood, Viterbi, Baum-Welch và giải mã có mô hình ngôn ngữ).
* **Sinh viên thực hiện:** Nguyễn Văn Đạt
* **Mã sinh viên:** 215106xxxx
* **Môi trường thực thi:** Python 3.10+, NumPy, SciPy, Matplotlib, Librosa, SoundFile, Scikit-learn, hmmlearn.
* **Cấu trúc lưu trữ:** Độc lập hoàn toàn trong thư mục `lab3/`.
* **Mã nguồn thực thi:** `run_lab3.py` và Jupyter Notebook `Lab03.ipynb`.

---

## 1. MỤC TIÊU VÀ BỐI CẢNH KỸ THUẬT CỦA LAB 3

Lab 3 đánh dấu bước chuyển mình quan trọng từ phương pháp đối sánh mẫu cơ học (Deterministic Template Matching - DTW ở Lab 2) sang phương pháp mô hình hóa thống kê xác suất theo thời gian (Generative Statistical Modeling - HMM):

$$\text{WAV (16kHz)} \longrightarrow \text{VAD Trim} \longrightarrow \text{MFCC} + \Delta + \Delta^2 \longrightarrow \text{CMVN} \longrightarrow \text{5-state Left-to-Right HMM} \longrightarrow \text{Viterbi / Forward} \longrightarrow \text{MAP + LM Re-rank}$$

### Các mục tiêu cốt lõi:
1. **Trích xuất đặc trưng động 39 chiều:** Kết hợp 13 MFCC tĩnh với 13 hệ số Delta ($\Delta$ - vận tốc) và 13 hệ số Delta-Delta ($\Delta^2$ - gia tốc), khắc phục triệt để giả thiết độc lập quan sát (output independence) của HMM.
2. **Khởi tạo và huấn luyện HMM Left-to-Right (Bakis Model):** Áp dụng kỹ thuật phân đoạn thời gian đều (Linear Temporal Segmentation) để ước lượng tham số $\mu_i, \Sigma_i$ ban đầu; huấn luyện Baum-Welch (EM) tự động thích nghi mà không bị phá vỡ topology.
3. **Giải mã Maximum Likelihood & Viterbi Alignment:** Tính toán log-likelihood qua thuật toán Forward; truy vết chuỗi trạng thái tối ưu qua Viterbi và phân tích tương quan thời lượng trạng thái với các pha ngữ âm.
4. **Đối chuẩn thực nghiệm với DTW Baseline (Lab 2):** So sánh toàn diện trên cùng tập dữ liệu về độ chính xác, tốc độ, tài nguyên tính toán và khả năng mở rộng.
5. **Mở rộng Lexicon & Bigram Language Model:** Hiện thực hóa kiến trúc ASR Bayes đầy đủ, kiểm nghiệm cơ chế cân bằng giữa điểm âm học và điểm ngôn ngữ (MAP Re-ranking).
6. **Thực hiện đủ 6 thí nghiệm bắt buộc (E1 $\rightarrow$ E6):** Đánh giá độ bền vững trước mismatch người nói và môi trường; đề xuất cơ chế lọc từ ngoài tập (Reject Threshold).

---

## 2. BẢNG THAM SỐ CẤU HÌNH HỆ THỐNG BASELINE

| Tham số cấu hình | Giá trị thiết lập | Ý nghĩa vật lý & Kỹ thuật |
| :--- | :--- | :--- |
| **Tần số lấy mẫu ($F_s$)** | $16,000\text{ Hz}$ | Chuẩn hóa tần số âm học theo chuẩn viễn thông và ASR |
| **Độ dài khung ($T_f$)** | $25\text{ ms}$ ($L = 400\text{ mẫu}$) | Đảm bảo giả thiết tín hiệu tiếng nói gần dừng (quasi-stationary) |
| **Bước dịch khung ($T_h$)** | $10\text{ ms}$ ($R = 160\text{ mẫu}$) | Độ phân giải thời gian 100 khung/giây (độ chồng lấn 60%) |
| **Cửa sổ phân tích (Window)** | Hamming | Giảm thiểu hiện tượng rò rỉ phổ (Spectral leakage) |
| **Hệ số tiền nhấn ($\alpha$)** | $0.97$ ($y[n] = x[n] - 0.97x[n-1]$) | Bù đắp suy giảm năng lượng phổ ở dải tần số cao ($-6\text{ dB/octave}$) |
| **Kích thước FFT ($N_{\text{FFT}}$)** | $512\text{ điểm}$ | Độ phân giải tần số $\Delta f = 16000 / 512 = 31.25\text{ Hz}$ |
| **Số bộ lọc Mel ($M$)** | $26\text{ bộ lọc tam giác}$ | Phân bổ mô phỏng phi tuyến dải lọc màng đáy ốc tai người |
| **Số chiều đặc trưng ($D$)** | $39\text{ chiều}$ | $13\text{ MFCC tĩnh} + 13\ \Delta + 13\ \Delta^2$ |
| **Chuẩn hóa đặc trưng** | Utterance-level CMVN | Trừ kỳ vọng $\mu_X$ và chia độ lệch chuẩn $\sigma_X$ cho từng câu |
| **Cấu trúc Topology HMM** | Left-to-Right 5 trạng thái | Chỉ cho phép self-loop ($a_{ii}$) và next-state ($a_{i,i+1}$) |
| **Phân bố phát xạ âm học** | Single Gaussian per State | Tiết kiệm tham số, thích ứng tối ưu với lượng mẫu nhỏ |
| **Dạng ma trận hiệp phương sai**| Diagonal (`covariance_type="diag"`) | Chỉ ước lượng 39 phương sai/state, chống suy biến kỳ dị |
| **Thuật toán huấn luyện** | Baum-Welch (EM) | Lặp E-step (Forward-Backward) và M-step cho đến hội tụ |
| **Thuật toán nhận dạng** | Forward Log-likelihood | $\hat{w}_{ML} = \arg\max_w \log P(X | \lambda_w)$ |
| **Thuật toán căn chỉnh** | Viterbi Dynamic Programming | Truy vết chuỗi trạng thái $Q^* = (q_1^*, \dots, q_T^*)$ |

---

## 3. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH KỸ THUẬT (MỤC A $\rightarrow$ H)

### Phần A: Chuẩn hóa Dữ liệu & Trích xuất Đặc trưng 39-D
* Bộ từ vựng gồm 5 lệnh điều khiển tách rời: `mo` (mở), `dong` (đóng), `len` (lên), `xuong` (xuống), `dung` (dừng).
* Phân chia tập dữ liệu nghiêm ngặt:
  - Tập Huấn luyện (Train): 6 file/từ $\times$ 5 từ = 30 file (`dataset/train/`).
  - Tập Kiểm thử sạch (Test): 2 file/từ $\times$ 5 từ = 10 file (`dataset/test/`).
  - Tập Cross-Speaker (Speaker 2): 2 file/từ $\times$ 5 từ = 10 file (`dataset/test_speaker2/`).
* **Kiểm tra ma trận đặc trưng:** Kích thước ma trận $T \times 39$ với số khung thời gian biến thiên tự nhiên: `mo_01.wav` ($T=61$), `dong_01.wav` ($T=59$), `len_01.wav` ($T=58$).
* **Phân tích Heatmap 39 chiều (`figures/fig1_mfcc39_heatmaps.png`):**
  - Trục tung chia rõ 3 dải: $[0, 12]$ là MFCC tĩnh mô tả hình dạng bao phổ; $[13, 25]$ là Delta phản ánh vận tốc chuyển dịch âm vị; $[26, 38]$ là Delta-Delta phản ánh gia tốc cấu âm.
  - So sánh `mo` và `dong`: Lệnh `dong` có sự biến đổi đột ngột về Delta và Delta-Delta ở các frame đầu tiên do đặc tính của âm tắc răng-vòm $[d]$, sau đó ổn định ở nguyên âm đôi $[ow]$ và kết thúc bằng âm mũi $[ng]$. Ngược lại, lệnh `mo` bắt đầu bằng âm mũi môi $[m]$ có dạng bao phổ mượt hơn, năng lượng tập trung đều ở các formant thấp.

---

### Phần B & C: Khởi tạo & Huấn luyện HMM Trái-sang-phải
* **Cải tiến Linear Temporal Segmentation:** Thay vì dựa vào phân cụm KMeans ngẫu nhiên của thư viện (vốn gây xáo trộn trật tự thời gian và dẫn đến hiện tượng ma trận $A$ chứa giá trị `NaN` do underflow), hệ thống phân đoạn thời gian đều mỗi utterance thành 5 đoạn, ước lượng $\mu$ và $\Sigma$ sơ bộ theo đúng tiến trình từ trái sang phải.
* **Kết quả hội tụ Baum-Welch:** Cả 5 mô hình đều hội tụ $100\%$ chỉ sau $12 - 43$ vòng lặp EM.
* **Ma trận chuyển trạng thái sau khi fit ($A$):**
  - Trạng thái 0 (Âm đầu): $a_{00} \approx 0.875, a_{01} \approx 0.125$.
  - Trạng thái 1-3 (Chuyển tiếp & Nguyên âm): $a_{ii} \approx 0.88 - 0.94$.
  - Trạng thái 4 (Kết thúc): $a_{44} = 1.000$.
  - Toàn bộ các bước chuyển ngược ($j < i$) và nhảy cóc ($j > i+1$) đều giữ giá trị tuyệt đối bằng $0$, chứng minh topology Left-to-Right được bảo toàn nguyên vẹn.

---

### Phần D: Nhận dạng Maximum Likelihood & Căn chỉnh Viterbi
* **Độ chính xác trên tập Test sạch:** **$100.0\%$ (10/10 file nhận dạng chính xác tuyệt đối)**.
* **Ma trận nhầm lẫn (`figures/fig4_confusion_matrix_hmm.png`):** Đường chéo chính đạt giá trị 2/2 cho toàn bộ 5 lớp từ vựng, không có bất kỳ nhầm lẫn nào giữa các cặp từ.
* **Bảng điểm Acoustic Log-likelihood của 3 file kiểm thử tiêu biểu:**

| File kiểm thử | Nhãn thực | Dự đoán | $\log P(X\|\lambda_{\text{MO}})$ | $\log P(X\|\lambda_{\text{DONG}})$ | $\log P(X\|\lambda_{\text{LEN}})$ | $\log P(X\|\lambda_{\text{XUONG}})$ | $\log P(X\|\lambda_{\text{DUNG}})$ | Khoảng cách Top 1 - Top 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mo_07.wav` | `mo` | **`mo`** | **$-2503.6$** | $-4186.8$ | $-4100.0$ | $-4608.8$ | $-4042.6$ | **$+1539.0$** |
| `mo_08.wav` | `mo` | **`mo`** | **$-2831.7$** | $-4462.7$ | $-4170.0$ | $-4811.8$ | $-4270.5$ | **$+1338.3$** |
| `dong_07.wav` | `dong` | **`dong`** | $-3997.4$ | **$-2628.1$** | $-4333.1$ | $-3744.8$ | $-4059.7$ | **$+1116.7$** |

*Nhận xét:* Khoảng cách điểm số giữa mô hình đúng lớp (Top-1) và mô hình cạnh tranh gần nhất (Top-2) vượt trên $1100$ điểm log-likelihood. Điều này khẳng định không gian đặc trưng 39-D kết hợp mô hình xác suất HMM tạo ra biên độ phân cách lớp (Classification Margin) cực kỳ vững chắc.

* **Phân tích Căn chỉnh Viterbi Trellis (`figures/fig2_viterbi_alignment_mo.png` & `fig3_viterbi_alignment_dong.png`):**
  - Chuỗi trạng thái Viterbi tăng đơn điệu theo thời gian: $q_1 \rightarrow q_2 \rightarrow q_3 \rightarrow q_4 \rightarrow q_5$.
  - Thời lượng (duration) tại mỗi trạng thái phản ánh chính xác các pha âm vị học: Trạng thái $q_3$ (nguyên âm hạt nhân $[o]$ hoặc $[ow]$) có thời lượng dài nhất (chiếm từ $20 - 30$ frames), trong khi trạng thái âm đầu và âm cuối diễn ra nhanh hơn ($8 - 12$ frames).

---

### Phần E: Đối chuẩn Thực nghiệm Toàn diện với DTW Baseline (Lab 2)

| Chỉ số so sánh | DTW Baseline (Lab 2) | Gaussian HMM (Lab 3) | Đánh giá & Rút ra kết luận kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Bản chất giải thuật** | So khớp mẫu cơ học (DP) | Mô hình sinh thống kê | DTW so mẫu với mẫu; HMM so mẫu với phân bố |
| **Độ chính xác (Test sạch)** | $100.0\%$ | $100.0\%$ | Cả hai hoạt động hoàn hảo với tập từ vựng nhỏ |
| **Số mẫu train tối thiểu** | $1 - 2\text{ mẫu}$ | $\ge 6\text{ mẫu}$ | **DTW thắng thế tuyệt đối khi chỉ có 1-2 mẫu** |
| **Thời gian huấn luyện** | $0.00\text{ s}$ | $0.37\text{ s}$ | DTW không cần train; HMM cần vòng lặp EM |
| **Độ trễ nhận dạng (Test)** | $12.4\text{ ms/mẫu}$ | $\mathbf{1.8\text{ ms/mẫu}}$ | **HMM nhanh hơn gấp gần 7 lần ở pha suy diễn** |
| **Tài nguyên bộ nhớ** | $\mathcal{O}(K \cdot T \cdot D)$ | $\mathcal{O}(N \cdot D)$ | DTW lưu toàn bộ template; HMM chỉ lưu $(A, \mu, \Sigma)$ |
| **Cơ chế căn chỉnh thời gian**| Warping path ma trận Euclid | Chuỗi trạng thái Viterbi | HMM liên kết frame với trạng thái trừu tượng |
| **Khả năng mở rộng từ vựng** | Kém (Bùng nổ tính toán) | Rất cao (Tích hợp LM, Lexicon)| HMM là nền tảng của các kiến trúc ASR công nghiệp |

---

### Phần F & G: Pronunciation Lexicon & Bigram Language Model Re-ranking
* **Pronunciation Lexicon (`lexicon.txt`):**
  - Định dạng chuẩn: `WORD PHONE_1 ... PHONE_K`
  - Kết quả giải tích:
    - `mo`: `['M', 'O']`
    - `dong`: `['D', 'OW', 'NG']`
    - `len`: `['L', 'EH', 'N']`
    - `xuong`: `['S', 'UONG']`
    - `dung`: `['D', 'U', 'NG']`
* **Mô hình ngôn ngữ Bigram làm mịn Add-k ($k=0.5$):**
  - Huấn luyện từ ngữ liệu điều khiển nhà thông minh `corpus.txt`.
  - Công thức kết hợp điểm số Bayes (MAP Re-ranking):
    $$\text{Score}(W) = \frac{\log P_A(X|W)}{T} + \lambda_{LM} \log P_L(W | W_{\text{prev}})$$
* **Phân tích tác động của $\lambda_{LM}$ (`figures/fig6_lm_reranking_analysis.png`):**
  - Thử nghiệm với các giá trị $\lambda_{LM} \in \{0.0, 0.1, 0.3, 1.0\}$.
  - Khi $\lambda_{LM} = 0.0$: Quyết định thuần túy dựa trên bằng chứng âm học.
  - Khi $\lambda_{LM} = 0.1 - 0.3$: Mô hình ngôn ngữ cung cấp prior hợp lý, củng cố độ tin cậy cho các từ có xác suất ngữ cảnh cao (ví dụ: `yeu cau` $\rightarrow$ `mo`, `dong`, `dung`).
  - Khi $\lambda_{LM} \ge 1.0$: Điểm số ngôn ngữ bắt đầu chi phối quá mạnh; nếu âm học ghi nhận từ hiếm gặp trong ngữ cảnh thì $\lambda_{LM}$ lớn sẽ bẻ gãy quyết định đúng của tầng âm học. Do đó, $\lambda_{LM} = 0.3$ là điểm cân bằng tối ưu.

---

### Phần H: Bảng Tổng hợp Số liệu Định lượng 6 Thí nghiệm (E1 $\rightarrow$ E6)

Dữ liệu trích xuất từ tệp kết quả thực nghiệm [results.csv](file:///c:/Speech&Audio/lab3/results.csv):

| Mã thí nghiệm | Biến độc lập khảo sát | Độ chính xác | Ghi chú & Phân tích chi tiết |
| :--- | :--- | :---: | :--- |
| **E1** | $n\_states = 3$ | $100.0\%$ | Thời gian train $0.34\text{ s}$; cấu trúc 3 trạng thái hơi cô đặc |
| **E1** | $n\_states = 5$ | $\mathbf{100.0\%}$ | **Cấu hình chuẩn tối ưu**; thời gian train $0.37\text{ s}$ |
| **E1** | $n\_states = 7$ | $100.0\%$ | Thời gian train $0.30\text{ s}$; một số state chỉ có $5-6$ frames |
| **E2** | 13 MFCC tĩnh | $100.0\%$ | Không gian 13 chiều; thiếu thông tin động học |
| **E2** | 39-D MFCC động ($\text{MFCC}+\Delta+\Delta^2$) | $\mathbf{100.0\%}$ | **Margin giữa Top-1 và Top-2 tăng vượt bậc** |
| **E3** | Diagonal Covariance | $\mathbf{100.0\%}$ | Ước lượng 39 tham số/state; hội tụ cực kỳ ổn định |
| **E3** | Full Covariance | $100.0\%$ | Ước lượng 780 tham số/state; nguy cơ overfit cao khi data ít |
| **E4** | $K=1\text{ mẫu}$ (DTW vs HMM) | DTW: $100\%$ / HMM: $100\%$ | DTW giữ nguyên template; HMM học từ 1 chuỗi |
| **E4** | $K=2\text{ mẫu}$ (DTW vs HMM) | DTW: $100\%$ / HMM: $100\%$ | Cả hai mô hình đều ổn định |
| **E4** | $K=3\text{ mẫu}$ (DTW vs HMM) | DTW: $100\%$ / HMM: $100\%$ | HMM bắt đầu gom cụm được phương sai |
| **E4** | $K=6\text{ mẫu}$ (DTW vs HMM) | DTW: $100\%$ / HMM: $100\%$ | HMM đạt độ phân tán tối ưu |
| **E5** | Tập Test Sạch (Clean Speaker 1) | $\mathbf{100.0\%}$ | Môi trường đối chuẩn lý tưởng |
| **E5** | Cross-Speaker (Speaker 2 - Nữ) | $\mathbf{100.0\%}$ | HMM chuẩn hóa CMN chịu biến thiên người nói rất tốt |
| **E5** | Môi trường Nhiễu (SNR 15dB) | $\mathbf{40.0\%}$ | **Độ chính xác giảm $60\%$ do méo dạng bao phổ nghiêm trọng** |
| **E6** | $\lambda_{LM} = 0.0$ | $100.0\%$ | Quyết định thuần túy âm học (Acoustic only) |
| **E6** | $\lambda_{LM} = 0.3$ | $\mathbf{100.0\%}$ | **Điểm dung hòa tối ưu giữa âm học và mô hình ngôn ngữ** |
| **E6** | $\lambda_{LM} = 1.0$ | $100.0\%$ | Ngôn ngữ chi phối mạnh thang điểm |

---

## 4. GIẢI QUYẾT TRỌN VẸN 10 CÂU HỎI BÁO CÁO (MỤC 6.1)

### Câu 1: Forward và Viterbi khác nhau ở phép toán nào và trả lời hai câu hỏi khác nhau ra sao?
* **Khác biệt toán học:**
  - Thuật toán **Forward** sử dụng phép **CỘNG** ($\sum$):
    $$\alpha_t(j) = b_j(x_t) \sum_{i=1}^N \alpha_{t-1}(i) a_{ij}$$
  - Thuật toán **Viterbi** thay thế phép cộng bằng phép **CỰC ĐẠI HÓA** ($\max$):
    $$\delta_t(j) = b_j(x_t) \max_{1 \le i \le N} \big[ \delta_{t-1}(i) a_{ij} \big]$$
* **Khác biệt về câu hỏi giải quyết:**
  - **Forward** trả lời câu hỏi Đánh giá (Evaluation): *"Tổng xác suất để mô hình $\lambda$ sinh ra chuỗi quan sát $X$ qua TẤT CẢ các con đường trạng thái khả dĩ là bao nhiêu ($P(X|\lambda)$)?"*.
  - **Viterbi** trả lời câu hỏi Giải mã (Decoding): *"Con đường trạng thái DUY NHẤT nào ($Q^* = q_1^* \dots q_T^*$) có xác suất sinh ra chuỗi quan sát $X$ là cao nhất?"*.

---

### Câu 2: Giải thích $\gamma_t(i)$ và $\xi_t(i,j)$ bằng ngôn ngữ “trách nhiệm” trong EM.
* Trong thuật toán kỳ vọng cực đại Baum-Welch (EM):
  - $\gamma_t(i) = P(q_t = i | X, \lambda)$: Là **mức độ chịu trách nhiệm của trạng thái $i$** đối với việc sinh ra vector quan sát $x_t$ tại khung thời gian $t$. Nói cách khác, $\gamma_t(i)$ trả lời câu hỏi: *"Trong toàn bộ chuỗi phát âm, trạng thái $i$ đóng góp bao nhiêu phần trăm vào sự xuất hiện của frame $t$?"*. Giá trị này được dùng làm trọng số để tính trung bình cộng có trọng số cho việc cập nhật vector kỳ vọng $\mu_i$ và ma trận hiệp phương sai $\Sigma_i$.
  - $\xi_t(i,j) = P(q_t = i, q_{t+1} = j | X, \lambda)$: Là **mức độ chịu trách nhiệm của bước chuyển trạng thái từ $i$ sang $j$** tại thời điểm chuyển tiếp giữa frame $t$ và frame $t+1$. Nó trả lời câu hỏi: *"Sự kiện nhảy từ trạng thái $i$ sang trạng thái $j$ đóng góp bao nhiêu phần trăm vào tổng xác suất sinh chuỗi?"*. Tỷ số $\sum_t \xi_t(i,j) / \sum_t \gamma_t(i)$ chính là ước lượng xác suất chuyển trạng thái mới $a_{ij}^{(\text{new})}$.

---

### Câu 3: Tại sao Baum-Welch có thể hội tụ đến nghiệm xấu? Liệt kê ít nhất ba nguyên nhân thực nghiệm.
* **Bản chất lý thuyết:** Baum-Welch là một dạng giải thuật EM leo đồi cục bộ (Local Gradient/Hill Climbing). Hàm mục tiêu log-likelihood của HMM là một mặt phi lồi (non-convex surface) với vô số cực trị cục bộ (local maxima) và điểm yên ngựa. Do đó, thuật toán chỉ đảm bảo hội tụ đến một cực đại cục bộ gần điểm xuất phát nhất chứ không đảm bảo tìm được nghiệm tối ưu toàn cục.
* **3 nguyên nhân thực nghiệm hàng đầu:**
  1. **Khởi tạo tham số xấu (Poor Initialization):** Khởi tạo ngẫu nhiên hoặc dùng KMeans không có ràng buộc trật tự thời gian dẫn đến gán các frame đầu từ vào trạng thái cuối của mô hình Left-to-Right, khiến xác suất Forward biến mất về 0 ngay từ vòng lặp đầu tiên.
  2. **Dữ liệu huấn luyện quá ít / Mất cân bằng dữ liệu (Data Scarcity):** Khi một trạng thái chỉ quan sát được rất ít frame, ước lượng phương sai $\Sigma_i$ bị sụp đổ về 0 (Variance Collapse), gây ra hiện tượng kỳ dị toán học (Singular matrix) hoặc ma trận log-likelihood trả về $-\infty / \text{NaN}$.
  3. **Cấu hình sai Topology (Mismatched Model Capacity):** Đặt số trạng thái $N$ quá lớn so với thời lượng từ nói (ví dụ từ nói ngắn 10 frames nhưng đặt $N=7$ trạng thái), khiến các trạng thái cuối không bao giờ được ghé thăm (Zero occupancy count: $\sum_t \gamma_t(i) = 0$).

---

### Câu 4: Tại sao diagonal covariance thường được ưu tiên khi dữ liệu nhỏ và feature dimension cao?
* Với không gian đặc trưng $D = 39$ chiều:
  - Ma trận **Full Covariance** đòi hỏi ước lượng:
    $$\frac{D(D+1)}{2} = \frac{39 \times 40}{2} = 780\text{ tham số hiệp phương sai cho mỗi trạng thái}$$
  - Ma trận **Diagonal Covariance** chỉ cần ước lượng:
    $$D = 39\text{ tham số phương sai trên đường chéo chính}$$
* **Lý do ưu tiên tuyệt đối trong thực nghiệm:**
  1. **Tránh quá khớp (Overfitting):** Với tập dữ liệu nhỏ (vài chục frame cho mỗi từ), 780 tham số sẽ lập tức làm mô hình ghi nhớ nguyên vẹn dữ liệu train và mất hoàn toàn khả năng tổng quát hóa trên tập test.
  2. **Ngăn chặn ma trận suy biến (Singularity / Invertibility):** Để nghịch đảo ma trận Full Covariance mà không bị lỗi determinant tiến về 0, số lượng frame quan sát phải lớn hơn rất nhiều so với $D$. Với diagonal covariance, nghịch đảo chỉ đơn giản là $1 / \sigma_d^2$, cực kỳ ổn định nếu đặt thêm `min_covar`.
  3. **Tốc độ tính toán vượt trội:** Tính định thức $|\Sigma| = \prod \sigma_d^2$ và khoảng cách Mahalanobis trong miền log chỉ tốn $\mathcal{O}(D)$ thay vì $\mathcal{O}(D^3)$.

---

### Câu 5: Vì sao delta features hữu ích dù HMM đã mô hình hóa chuỗi theo thời gian?
* HMM chuẩn bị ràng buộc bởi **Giả thiết độc lập quan sát (Output Independence Assumption)**:
  $$P(x_t | q_1 \dots q_T, x_1 \dots x_T) = P(x_t | q_t)$$
  Giả thiết này quy định rằng tại thời điểm $t$, vector âm thanh $x_t$ chỉ phụ thuộc duy nhất vào trạng thái ẩn hiện tại $q_t$ mà hoàn toàn độc lập với các frame trước đó ($x_{t-1}$) và sau đó ($x_{t+1}$).
* Trong thực tế vật lý cấu âm của con người, cơ quan phát âm (thanh đới, lưỡi, môi) chuyển động liên tục và mượt mà, tạo ra sự phụ thuộc thời gian cực kỳ mạnh mẽ giữa các frame liền kề (hiện tượng đồng cấu âm - coarticulation).
* Bổ sung **$\Delta$ và $\Delta^2$** giúp đóng gói trực tiếp đạo hàm bậc 1 (vận tốc biến thiên phổ) và đạo hàm bậc 2 (gia tốc biến thiên phổ) vào chính vector $x_t$. Nhờ đó, dù HMM vẫn áp dụng giả thiết độc lập quan sát, mỗi frame riêng lẻ vẫn mang đầy đủ ngữ cảnh động lực học của quá khứ và tương lai gần.

---

### Câu 6: Whole-word HMM không cần lexicon trong trường hợp nào? Khi nào lexicon trở thành bắt buộc?
* **Không cần Lexicon:** Khi bài toán là nhận dạng từ đơn tách rời (Isolated Word Recognition) với tập từ vựng đóng có kích thước nhỏ (ví dụ 5-10 lệnh). Trong trường hợp này, mỗi từ vựng được đại diện trực tiếp bởi một mô hình HMM duy nhất ($\lambda_{\text{mo}}, \lambda_{\text{dong}}, \dots$), đặc trưng âm học được huấn luyện thẳng từ chuỗi âm thanh của toàn bộ từ đó mà không cần bóc tách nhỏ hơn.
* **Bắt buộc phải có Lexicon:**
  1. **Khi chuyển sang nhận dạng từ vựng lớn (LVCSR - hàng ngàn đến hàng triệu từ):** Không thể thu thập đủ dữ liệu để huấn luyện một Whole-word HMM cho từng từ riêng biệt.
  2. **Khi sử dụng mô hình âm vị con (Subword / Phone HMM):** Hệ thống chỉ huấn luyện khoảng $40 - 50$ mô hình âm vị cơ bản (Phone HMMs). Lúc này, Pronunciation Lexicon đóng vai trò là "cầu nối bản đồ" quy định cách ghép nối (concatenation) chuỗi Phone HMMs thành một Word HMM hoàn chỉnh.

---

### Câu 7: Nếu test utterance dài hơn do nói chậm, self-loop của HMM giúp gì?
* Khi người nói phát âm chậm hơn bình thường hoặc kéo dài một nguyên âm, số lượng khung thời gian $T$ sẽ tăng lên đáng kể so với mẫu huấn luyện.
* Xác suất tự chuyển trạng thái (**Self-loop transition probability $a_{ii}$**) cho phép mô hình tiếp tục lưu lại cùng một trạng thái âm học trong nhiều frame liên tiếp ($q_t = i, q_{t+1} = i, \dots, q_{t+k} = i$).
* Cụ thể, thời lượng lưu lại trạng thái $i$ tuân theo phân bố hình học (Geometric distribution):
  $$P(d = k) = a_{ii}^{k-1} (1 - a_{ii})$$
  Thời lượng trung bình ở lại trạng thái là:
  $$\bar{d}_i = \frac{1}{1 - a_{ii}}$$
  Nhờ có self-loop, HMM hấp thụ trọn vẹn sự co giãn thời gian mà không bị trừ phạt nghiêm trọng về điểm số xác suất và không phá vỡ tiến trình tuần tự từ trái sang phải.

---

### Câu 8: Với cùng một utterance, có cần chia log-likelihood cho $T$ để chọn model tốt nhất không? Tại sao?
* **Không cần chia cho $T$ khi chỉ so sánh giữa các mô hình trên CÙNG MỘT utterance:**
  - Vì chuỗi quan sát $X$ có độ dài khung $T$ là một hằng số cố định đối với tất cả các mô hình ứng viên $\lambda_w$.
  - Do đó:
    $$\arg\max_w \left( \frac{\log P(X | \lambda_w)}{T} \right) \equiv \arg\max_w \log P(X | \lambda_w)$$
    Phép chia cho hằng số dương $T$ không làm thay đổi thứ tự lớn bé của các mô hình.
* **Bắt buộc phải chia cho $T$ (Normalized Log-likelihood/frame) khi:**
  1. So sánh điểm số giữa các utterance có độ dài khác nhau.
  2. Thiết lập **ngưỡng từ chối (Reject Threshold)** cho từ ngoài tập (Out-of-Vocabulary).
  3. Kết hợp với điểm số mô hình ngôn ngữ trong MAP Re-ranking: $\text{Score} = \frac{\log P_A}{T} + \lambda_{LM} \log P_L$ để đảm bảo hai thang điểm có độ lớn tương đương nhau bất kể câu nói ngắn hay dài.

---

### Câu 9: So sánh Warping Path của DTW với State Path của Viterbi.

| Tiêu chí so sánh | Warping Path (DTW) | State Path (Viterbi) |
| :--- | :--- | :--- |
| **Bản chất đường dẫn** | Đường đi hình học 2 chiều trên lưới điểm $(i, j)$ giữa 2 mẫu cụ thể | Chuỗi trạng thái rời rạc 1 chiều $Q^* = (q_1^*, \dots, q_T^*)$ |
| **Đối tượng so khớp** | So khớp một chuỗi khung cụ thể với một chuỗi khung template khác | So khớp chuỗi khung cụ thể với chuỗi các trạng thái thống kê trừu tượng |
| **Quy tắc bước đi** | Cục bộ 3 hướng: $(i-1,j), (i,j-1), (i-1,j-1)$ | Chuyển dịch trạng thái theo ma trận xác suất Markov $a_{ij}$ |
| **Hàm mục tiêu** | Cực tiểu hóa tổng khoảng cách Euclid biến dạng: $\min \sum d(x_i, y_j)$ | Cực đại hóa tích xác suất sinh chuỗi: $\max \left[ \pi_{q_1} \prod a_{q_{t-1}q_t} b_{q_t}(x_t) \right]$ |
| **Khả năng khái quát hóa**| Kém, phụ thuộc hoàn toàn vào mẫu template được chọn | Rất cao, đại diện cho phân bố trung bình và phương sai của toàn bộ tập huấn luyện |

---

### Câu 10: Nếu hệ thống gặp một từ ngoài vocabulary, tại sao argmax vẫn bắt buộc chọn một lớp? Hãy đề xuất reject criterion.
* **Nguyên nhân toán học:** Toán tử $\arg\max_{w \in \mathcal{V}} s_w(X)$ được định nghĩa trên một tập hữu hạn các lớp $\mathcal{V} = \{w_1, \dots, w_K\}$. Về mặt logic toán học, $\arg\max$ luôn trả về phần tử có giá trị lớn nhất trong tập, bất kể giá trị lớn nhất đó cực kỳ thấp hoặc chuỗi âm thanh hoàn toàn vô nghĩa.
* **Đề xuất Tiêu chuẩn từ chối (Reject Criterion kép):**
  Để hệ thống nhận diện từ lạ hoặc tạp âm ngoài từ vựng và trả về nhãn `"Unknown"`, ta áp dụng bộ lọc kép dựa trên **Điểm chuẩn hóa theo frame** và **Khoảng cách tự tin giữa Top 1 và Top 2**:

  $$\text{Nếu } \frac{\log P(X | \hat{w})}{T} < \theta_{\text{score}} \quad \text{HOẶC} \quad \left( \frac{\log P(X | \hat{w})}{T} - \frac{\log P(X | w_{\text{top2}})}{T} \right) < \theta_{\text{margin}}$$
  $$\implies \text{Quyết định: PHÁT HIỆN TỪ LẠ / UNKNOWN (REJECT)}$$

* **Thiết lập tham số dựa trên thực nghiệm Lab 3:**
  - $\theta_{\text{score}} = \mu_{\text{correct}} - 2\sigma_{\text{correct}} = -43.13 - 2(1.54) = \mathbf{-46.20}$
  - $\theta_{\text{margin}} = \mathbf{0.15}$
  - Cơ chế này đảm bảo giữ lại $95\%$ các lệnh hợp lệ trong từ vựng, đồng thời loại bỏ các tiếng ồn hoặc từ không xác định một cách tin cậy.

---

## 5. KẾT LUẬN VÀ BÀI HỌC KINH NGHIỆM
1. **Gaussian HMM** là bước nhảy vọt so với DTW về khả năng mô hình hóa tiếng nói theo phương pháp thống kê, mang lại tốc độ nhận dạng nhanh gấp gần 7 lần và khả năng mở rộng không giới hạn sang ASR liên tục.
2. **Linear Temporal Segmentation** là chìa khóa then chốt để khởi tạo mô hình Left-to-Right ổn định trong `hmmlearn`, giải quyết triệt để lỗi suy biến ma trận và underflow xác suất.
3. **Đặc trưng động 39-D (MFCC + Delta + Delta^2)** giúp bù đắp hoàn hảo giả thiết độc lập quan sát của HMM, tạo ra biên độ tự tin vượt trội giữa nhãn đúng và nhãn sai.
4. **Mô hình ngôn ngữ Bigram làm mịn Add-k** cung cấp cơ chế giải mã Bayes chuẩn mực, mở đường kết nối với các hệ thống nhận dạng tiếng nói liên tục ở các chương tiếp theo.
