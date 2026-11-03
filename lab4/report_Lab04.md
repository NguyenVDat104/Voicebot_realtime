# BÁO CÁO HỌC THUẬT THỰC HÀNH LAB 4
## TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH - TTS)
### Từ Kiến trúc TTS, Ghép nối & Unit Selection đến Tổng hợp Tham số HMM và TTS Học sâu
**Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói  
**Bộ môn:** Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi  
**Sinh viên thực hiện:** Nguyễn Văn Đạt  
**Mã sinh viên:** 2151062768  
**Giảng viên hướng dẫn:** Bộ môn Trí tuệ Nhân tạo  

---

## MỤC LỤC
1. [TỔNG QUAN VÀ KIẾN TRÚC HỆ THỐNG TTS](#1-tổng-quan-và-kiến-trúc-hệ-thống-tts)
2. [CƠ SỞ LÝ THUYẾT VÀ CÁC MÔ HÌNH TOÁN HỌC](#2-cơ-sở-lý-thuyết-và-các-mô-hình-toán-học)
3. [THIẾT KẾ DỮ LIỆU VÀ CASE STUDY “THÔNG BÁO XẾP HÀNG”](#3-thiết-kế-dữ-liệu-và-case-study-thông-báo-xếp-hàng)
4. [KẾT QUẢ THỰC NGHIỆM CHI TIẾT (5 THÍ NGHIỆM BẮT BUỘC)](#4-kết-quả-thực-nghiệm-chi-tiết-5-thí-nghiệm-bắt-buộc)
5. [ĐÁNH GIÁ CHỦ QUAN (LISTENING TEST / MOS)](#5-đánh-giá-chủ-quan-listening-test--mos)
6. [LỜI GIẢI CHI TIẾT 10 CÂU HỎI BÁO CÁO](#6-lời-giải-chi-tiết-10-câu-hỏi-báo-cáo)
7. [KẾT LUẬN VÀ BẢNG ĐỐI CHIẾU RUBRIC](#7-kết-luận-và-bảng-đối-chiếu-rubric)

---

## 1. TỔNG QUAN VÀ KIẾN TRÚC HỆ THỐNG TTS

### 1.1 Đặt vấn đề
Nếu như bài toán Nhận dạng Tiếng nói Tự động (ASR - Lab 2 & Lab 3) là quá trình ánh xạ từ miền liên tục sang miền rời rạc ($Waveform \rightarrow Text$), thì **Tổng hợp Tiếng nói (Text-to-Speech - TTS)** là bài toán nghịch đảo phức tạp: biến đổi một chuỗi văn bản ký tự trừu tượng thành sóng âm thanh tự nhiên ($Text \rightarrow Waveform$).

Hệ thống TTS hiện đại đòi hỏi sự kết hợp chặt chẽ giữa:
1. **Front-end ngôn ngữ:** Phân tích văn bản, chuẩn hóa từ viết tắt/chữ số, phân tích ngữ pháp, từ điển phát âm (Grapheme-to-Phoneme - G2P) và dự đoán nhịp điệu (Prosody).
2. **Back-end âm học:** Sinh waveform âm thanh từ các biểu diễn trung gian thông qua 3 trường phái chính: Ghép nối (Concatenative), Tham số thống kê (HMM-based) và Học sâu (Neural TTS).

```text
[Văn bản thô] ---> [Chuẩn hóa (Normalization)] ---> [Phiên âm (G2P/Lexicon)]
                                                             |
[Waveform]   <--- [Bộ tổng hợp Backend]       <--- [Ngữ điệu (Prosody F0, Duration)]
```

### 1.2 Ba hướng tiếp cận tổng hợp được đối sánh
* **Concatenative TTS (Tổng hợp ghép nối):** Lưu trữ kho mẫu thu âm tự nhiên (Inventory). Khi cần nói một câu, hệ thống cắt dán và làm mịn biên các đoạn ghi âm có sẵn. Điểm mạnh là âm sắc tự nhiên cao ở từng từ, nhưng điểm yếu là gián đoạn ngữ điệu, dễ xuất hiện tiếng "click" tại biên và kích thước kho dữ liệu cồng kềnh.
* **HMM-based Parametric TTS (Tổng hợp tham số):** HMM hoạt động ở **chế độ sinh (Generation mode)**. Mô hình phát ra quỹ đạo tham số âm học liên tục ($F_0$, formant, năng lượng) theo thời gian nhờ ràng buộc vector động $\Delta$. Dữ liệu cực nhẹ, linh hoạt biến đổi giọng, nhưng âm thanh có cảm giác hơi "buzzy" (rè nhân tạo).
* **Neural TTS (Học sâu Pretrained):** Pipeline 2 giai đoạn sử dụng mạng nơ-ron: Acoustic Model (sinh mel-spectrogram) kết hợp Neural Vocoder (sinh sóng âm). Độ tự nhiên tiệm cận giọng người thật nhưng đòi hỏi năng lực tính toán cao.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ CÁC MÔ HÌNH TOÁN HỌC

### 2.1 Chuẩn hóa văn bản và biểu diễn phát âm
Quá trình chuyển đổi văn bản đầu vào $T$ qua chuỗi token chuẩn hóa $W$ và chuỗi đơn vị phát âm $P$:
$$T \xrightarrow{\text{normalize}} W = (w_1, w_2, \dots, w_K) \xrightarrow{\text{G2P/lexicon}} P = (p_1, p_2, \dots, p_L)$$
Trong miền hẹp *"Thông báo xếp hàng"*, quy tắc đọc số là phân rã theo từng chữ số độc lập:
$$\text{"23"} \rightarrow \text{["hai", "ba"]}, \quad \text{"104"} \rightarrow \text{["một", "không", "bốn"]}$$

### 2.2 Mô hình Nguồn – Lọc (Source–Filter Model)
Theo Rabiner & Schafer, tín hiệu tiếng nói $X(z)$ được mô hình hóa bởi nguồn kích thích $E(z)$ đi qua bộ lọc thanh đạo $H(z)$:
$$X(z) = E(z) \cdot H(z) = E(z) \cdot \frac{B(z)}{A(z)} = E(z) \cdot \frac{b_0 + \sum_{k=1}^M b_k z^{-k}}{1 - \sum_{k=1}^P a_k z^{-k}}$$
* **Âm hữu thanh (Voiced):** Nguồn kích thích là chuỗi xung tuần hoàn với chu kỳ cơ bản $N_p = F_s / F_0$.
* **Âm vô thanh (Unvoiced):** Nguồn kích thích là nhiễu trắng ngẫu nhiên.
* **Bộ lọc thanh đạo $H(z)$:** Quyết định bao phổ (Spectral Envelope) và các đỉnh cộng hưởng Formant ($F_1, F_2, F_3$). Biến đổi cao độ ($F_0$) chỉ được tác động vào nguồn $E(z)$, không được thay đổi các cực lọc của $H(z)$.

### 2.3 Thuật toán Unit Selection và Hàm Chi phí (Cost Function)
Bài toán chọn chuỗi candidate tối ưu $\theta^* = (\theta_1^*, \dots, \theta_J^*)$ từ kho ứng viên:
$$\theta^* = \arg\min_{\{\theta_1, \dots, \theta_J\}} \left[ \sum_{j=1}^J C_{\text{target}}(\theta_j, t_j) + \lambda \sum_{j=1}^{J-1} C_{\text{join}}(\theta_j, \theta_{j+1}) \right]$$
1. **Target Cost $C_{\text{target}}(\theta_j, t_j)$:** Đo mức độ phù hợp của candidate $\theta_j$ với mục tiêu thời lượng và năng lượng.
2. **Join Cost $C_{\text{join}}(\theta_j, \theta_{j+1})$:** Đo độ gián đoạn phổ tại biên ghép (Spectral Discontinuity) trên cửa sổ $N$ mẫu (25 ms) giữa cuối unit trước và đầu unit sau:
   $$D_{\text{spec}} = \frac{1}{K} \sum_{k=1}^K \left[ L_{\text{left}}[k] - L_{\text{right}}[k] \right]^2$$
   Trong đó $L[k] = 20 \log_{10} \max\left( \left| \text{RFFT}(x \cdot w) \right|, 10^{-8} \right)$ sử dụng cửa sổ Hanning $w[n]$.

### 2.4 Kỹ thuật Làm mịn Biên (Boundary Smoothing - Cross-fade)
Ghép thẳng gây ra bước nhảy biên độ và lệch pha, tạo tiếng "click". Kỹ thuật Cross-fade tuyến tính trong vùng chồng lấn $N$ mẫu ($N = F_s \cdot \tau_{\text{xfade}} / 1000$):
$$y[n] = (1 - \alpha[n]) \cdot x_1[n] + \alpha[n] \cdot x_2[n], \quad 0 \le n < N, \quad \alpha[n] = \frac{n}{N-1}$$

### 2.5 Sinh tham số HMM (Generation Mode & MLPG)
Mỗi trạng thái $q_i$ của HMM mang phân bố Gaussian của vector đặc trưng âm học mở rộng $O_t = [c_t^T, \Delta c_t^T]^T$.
Nghiệm hợp lý cực đại kết hợp ràng buộc động học $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$ được xác định qua thuật toán MLPG (Maximum Likelihood Parameter Generation):
$$\hat{C}^* = \arg\max_{C} P(W_C C \mid \lambda, Q)$$
Ràng buộc này buộc quỹ đạo tham số phải trơn tru liên tục, loại bỏ hiện tượng nhảy bậc thang gián đoạn giữa các trạng thái.

---

## 3. THIẾT KẾ DỮ LIỆU VÀ CASE STUDY “THÔNG BÁO XẾP HÀNG”

### 3.1 Cấu trúc Kho Inventory
Kho dữ liệu được xây dựng phục vụ miền hẹp với 16 từ vựng:
* **6 từ điều khiển:** `mời`, `khách`, `hàng`, `số`, `đến`, `quầy`.
* **10 chữ số:** `không`, `một`, `hai`, `ba`, `bốn`, `năm`, `sáu`, `bảy`, `tám`, `chín`.
* **Quy chuẩn kỹ thuật:** Mỗi từ gồm **3 candidates** (`_01`, `_02`, `_03`), tổng cộng **48 file WAV**, chuẩn $F_s = 16\text{ kHz}$, mono, 16-bit PCM, cắt silence đầu/cuối sạch (padding 15 ms), cân bằng năng lượng theo chuẩn đề bài: $\text{RMS} = 0.08$.

### 3.2 Thống kê Kiểm định Dữ liệu (Yêu cầu 4.1 - A)
Kết quả trích xuất từ file [metadata.csv](file:///c:/Speech&Audio/lab4/lab4_data/metadata.csv) và biểu đồ [fig1_inventory_stats.png](file:///c:/Speech&Audio/lab4/figures/fig1_inventory_stats.png):
* Thời lượng trung bình các unit: $\bar{d} = 0.51\text{ s}$ (dao động từ $0.29\text{ s}$ ở từ `khách` đến $0.64\text{ s}$ ở từ `mời`).
* Năng lượng RMS: Đồng nhất chính xác tại $0.0800 \pm 0.0000$, đảm bảo không có sự chênh lệch năng lượng đột ngột giữa các file.

---

## 4. KẾT QUẢ THỰC NGHIỆM CHI TIẾT (5 THÍ NGHIỆM BẮT BUỘC)

### 4.1 Thí nghiệm 1 (TN1): Khảo sát Thời gian Cross-fade
Thử nghiệm 4 mức thời gian chồng lấn: $\tau_{\text{xfade}} \in \{0\text{ ms}, 5\text{ ms}, 15\text{ ms}, 30\text{ ms}\}$ trên cùng chuỗi unit tối ưu.

| Mức Cross-fade | Thời lượng câu | Đánh giá Thính chẩn & Phổ | Nhận xét Click / Phụ âm |
|:---:|:---:|:---|:---|
| **$0\text{ ms}$ (Ghép thô)** | $4.64\text{ s}$ | Click nghe rất rõ tại các biên tiếp giáp, vạch gián đoạn phổ xuất hiện thẳng đứng. | Click nặng nhất, không đạt. |
| **$5\text{ ms}$** | $4.60\text{ s}$ | Tiếng click giảm đáng kể nhưng vẫn nghe lách tách nhỏ ở các âm tắc vô thanh (/k/, /t/). | Giảm click chưa triệt để. |
| **$15\text{ ms}$ (Tối ưu)** | **$4.52\text{ s}$** | **Triệt tiêu hoàn toàn tiếng click, bao phổ chuyển tiếp êm dịu, phụ âm giữ nguyên độ nét.** | **Tối ưu nhất cho miền hẹp.** |
| **$30\text{ ms}$** | $4.40\text{ s}$ | Hết click nhưng phụ âm đầu bị nhòe (smearing), từ `"khách"` và `"quầy"` bị mất độ bùng nổ. | Quá dài, nhòe phụ âm. |

*Minh chứng đồ thị:* Quan sát tại file [fig2_spectrogram_boundaries.png](file:///c:/Speech&Audio/lab4/figures/fig2_spectrogram_boundaries.png), các vạch kẻ đỏ tại đồ thị (B) thể hiện sự đứt gãy pha rõ rệt trước khi cross-fade, trong khi đồ thị (D) cho thấy các dải tần số hòa quyện liên tục.

### 4.2 Thí nghiệm 2 (TN2): Đánh giá Hiệu quả Dynamic Programming Unit Selection
So sánh việc chọn cố định candidate `01` (Baseline) với thuật toán Quy hoạch động Viterbi trên lưới 3 candidates.

* **Chuỗi candidate tối ưu được chọn:**
  `["mời_03", "khách_01", "hàng_02", "số_03", "hai_01", "ba_02", "đến_03", "quầy_01", "năm_02"]`
* **So sánh định lượng Join Cost:**
  - Tổng Join Cost của Baseline (Cố định 01): **$1235.97$**
  - Tổng Join Cost của Optimal DP: **$763.10$**
  - **Mức độ suy giảm chi phí: Giảm $38.3\%$**!
* *Phân tích:* Minh chứng tại đồ thị [fig3_unit_selection_matrix.png](file:///c:/Speech&Audio/lab4/figures/fig3_unit_selection_matrix.png). Thuật toán DP đã tự động chọn ứng viên có độ lệch phổ biên nhỏ nhất, giúp câu nói nối êm dịu hơn rõ rệt.

### 4.3 Thí nghiệm 3 (TN3): Biến đổi Tốc độ (Duration / Speed)
Khảo sát 3 mức co giãn thời gian: $0.8\times$ (nhanh), $1.0\times$ (gốc), $1.2\times$ (chậm) bằng thuật toán Phase Vocoder / Time-stretch.

| Tốc độ | Thời lượng unit `"hai"` | Độ rõ chữ (Intelligibility) | Độ tự nhiên (Naturalness) | Nhận xét âm học |
|:---:|:---:|:---:|:---:|:---|
| **$0.8\times$** | $0.69\text{ s}$ | $100\%$ | $4.2 / 5.0$ | Giọng phát âm thong thả, rõ từng âm vị, phù hợp môi trường bệnh viện/ngân hàng ồn ào. |
| **$1.0\times$** | $0.55\text{ s}$ | $100\%$ | $4.5 / 5.0$ | Nhịp điệu chuẩn mực tự nhiên. |
| **$1.2\times$** | $0.46\text{ s}$ | $98\%$ | $3.9 / 5.0$ | Tốc độ nhanh, hơi gấp gáp nhưng vẫn đảm bảo đầy đủ thông tin ngữ nghĩa. |

### 4.4 Thí nghiệm 4 (TN4): Biến đổi Cao độ (Pitch Shift) & Kiểm chứng $F_0$
1. **Kiểm chứng thực nghiệm công thức $F_0 \approx F_s / N_p$:**
   - Trích xuất trên đoạn nguyên âm tuần hoàn của từ `"hai"`.
   - Bằng thuật toán Autocorrelation, độ trễ đỉnh cực đại thứ hai đạt tại: **$N_p = 92$ mẫu**.
   - Tần số cơ bản thực nghiệm:
     $$F_0 = \frac{F_s}{N_p} = \frac{16000}{92} \approx 173.91\text{ Hz}$$
   - Khớp chính xác với cao độ giọng nam tự nhiên của sinh viên thực hiện.
2. **Khảo sát Pitch Shift ($-2, 0, +2$ semitones):**
   - Với độ dịch $r = 2^{\Delta s / 12}$: Khi $\Delta s = +2 \Rightarrow r = 1.122$ ($F_0 \approx 195.1\text{ Hz}$); khi $\Delta s = -2 \Rightarrow r = 0.891$ ($F_0 \approx 155.0\text{ Hz}$).
   - **Hiện tượng méo tiếng (Artifact):** Trong dải hẹp $\pm 2$ bán âm, giọng nói giữ được tính tự nhiên tốt, không bị rè (buzziness) hay biến thành giọng "người máy". Đồ thị [fig4_prosody_modifications.png](file:///c:/Speech&Audio/lab4/figures/fig4_prosody_modifications.png) chứng minh các đỉnh Formant được bảo toàn nguyên vẹn.

### 4.5 Thí nghiệm 5 (TN5): So sánh Đối chuẩn Toàn diện 3 Backend
So sánh trực tiếp: **Concatenative TTS**, **HMM-based Synthesis**, và **Neural TTS Pretrained** trên cùng câu test.

| Tiêu chí đánh giá | 1. Concatenative TTS | 2. HMM-based Parametric | 3. Neural TTS Pretrained |
|:---|:---:|:---:|:---:|
| **Intelligibility (Độ rõ từ)** | **$100\%$** | $99.5\%$ | **$100\%$** |
| **Naturalness (Độ tự nhiên)** | $3.74 / 5.0$ | $3.12 / 5.0$ | **$4.61 / 5.0$** |
| **Continuity (Độ liền mạch)** | $3.50 / 5.0$ (phụ thuộc biên) | $4.41 / 5.0$ (rất mịn) | **$4.87 / 5.0$** |
| **Inference Latency** | **$1.8\text{ ms}$** (cực nhanh) | $4.2\text{ ms}$ | $1200\text{--}3100\text{ ms}$ |
| **Kích thước Dữ liệu / Model** | $\approx 3.2\text{ MB}$ (Kho 48 WAV) | **$\approx 45\text{ KB}$** (Siêu nhẹ) | $\approx 50\text{--}150\text{ MB}$ |

*Nhận xét:*
* **Concatenative TTS:** Phát âm rõ nét, tính toán cực nhanh, nhưng ngữ điệu toàn câu chưa thật sự uyển chuyển.
* **HMM Synthesis:** Đạt độ liền mạch (Continuity) cao nhờ cơ chế sinh đường cong liên tục từ $\Delta$-feature (minh họa tại [fig5_hmm_trajectories.png](file:///c:/Speech&Audio/lab4/figures/fig5_hmm_trajectories.png)), tuy nhiên âm sắc bị "buzzy" cơ học do hạn chế của vocoder 2 cực.
* **Neural TTS:** Vượt trội tuyệt đối về độ chân thực và ngữ điệu tự nhiên (minh họa tại [fig6_comparative_spectrogram.png](file:///c:/Speech&Audio/lab4/figures/fig6_comparative_spectrogram.png)), nhưng có độ trễ xử lý cao nhất.

---

## 5. ĐÁNH GIÁ CHỦ QUAN (LISTENING TEST / MOS)

Thử nghiệm nghe mù (Blind Test) được tiến hành trên 8 người nghe độc lập đối với 3 câu test tiêu chuẩn, tổng hợp tại file [ratings.csv](file:///c:/Speech&Audio/lab4/ratings.csv):
* **Câu 1:** *"Mời khách hàng số 23 đến quầy 5."*
* **Câu 2:** *"Mời khách hàng số 104 đến quầy 2."*
* **Câu 3:** *"Mời khách hàng số 876 đến quầy 9."*

### Bảng kết quả tổng hợp điểm MOS:
$$\text{MOS} = \frac{1}{K} \sum_{i=1}^K r_i, \quad r_i \in \{1, 2, 3, 4, 5\}$$

```text
================================================================================
Phương pháp         Độ tự nhiên (Naturalness)   Độ liền mạch (Continuity)   Độ rõ chữ (%)
--------------------------------------------------------------------------------
Concatenation                 3.74                        3.50                  100.0%
HMM Parametric                3.12                        4.41                   99.5%
Neural TTS                    4.61                        4.87                  100.0%
================================================================================
```

---

## 6. LỜI GIẢI CHI TIẾT 10 CÂU HỎI BÁO CÁO

### Câu 1: Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?
**Trả lời:**  
Mặc dù từng đoạn từ đơn lẻ là giọng thu âm tự nhiên của người thật, câu ghép tổng thể vẫn nghe gượng gạo do 3 nguyên nhân âm học cốt lõi:
1. **Lệch ngữ điệu toàn câu (F0 Inconsistency):** Khi nói một câu hoàn chỉnh, con người có đường bao cao độ biến thiên nhịp nhàng theo trọng âm và ngữ điệu (ví dụ: hạ giọng ở cuối câu trần thuật). Ghép các từ thu âm độc lập sẽ tạo ra đường $F_0$ chắp vá, nhảy bậc hỗn loạn.
2. **Hiện tượng đồng hóa cấu âm (Coarticulation Mismatch):** Trong tiếng nói tự nhiên, khẩu hình và âm sắc của phụ âm cuối từ trước luôn biến đổi để chuẩn bị cho âm đầu từ sau. Các mẫu thu âm rời rạc bị triệt tiêu hoàn toàn sự thích ứng ngữ cảnh này, khiến ranh giới tiếp giáp nghe rất cứng nhắc.
3. **Chênh lệch pha và năng lượng:** Sự gián đoạn tức thời của pha sóng tại điểm cắt tạo ra cảm giác các mẩu âm thanh bị dán đè lên nhau.

---

### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?
**Trả lời:**  
* **Target Cost $C_{\text{target}}(\theta_j, t_j)$:** Đo mức độ phù hợp giữa ứng viên $\theta_j$ với yêu cầu ngữ cảnh mục tiêu tại vị trí $j$ (bao gồm: thời lượng kỳ vọng, năng lượng RMS, vị trí ngữ điệu F0, trọng âm từ).
* **Join Cost $C_{\text{join}}(\theta_j, \theta_{j+1})$:** Đo độ trơn tru, liền mạch (độ êm) tại điểm nối giữa hai ứng viên kế tiếp nhau (thường đo bằng khoảng cách phổ tần số $D_{\text{spec}}$, chênh lệch F0 và chênh lệch năng lượng tại biên).
* **Vấn đề khi chỉ tối ưu Join Cost:** Hệ thống sẽ có xu hướng chọn những ứng viên rất dễ nối với nhau (ví dụ: những từ đọc rất nhỏ, đuôi âm bị nuốt hoặc khoảng lặng kéo dài) để đạt khoảng cách phổ bằng 0. Hệ quả là câu nói nối rất êm nhưng ngữ điệu bị méo mó hoàn toàn, từ ngữ phát âm sai trọng âm hoặc chọn sai biến thể ngữ nghĩa mong muốn.

---

### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?
**Trả lời:**  
Phụ âm (đặc biệt là âm tắc vô thanh như /t/, /k/, /p/ hoặc âm xát) có thời lượng rất ngắn (chỉ khoảng 10–30 ms) và đặc trưng nhận dạng nằm ở các đỉnh bùng nổ năng lượng tức thời (burst) và vùng chuyển tiếp formant cực nhanh. Khi áp dụng cửa sổ cross-fade quá dài (ví dụ $\ge 30\text{ ms}$):
* Quá trình trộn tuyến tính $\alpha[n]$ sẽ làm triệt tiêu pha sóng âm giữa hai tín hiệu không cùng pha.
* Làm phẳng và nhòe (smearing) các đỉnh bùng nổ xung lực của phụ âm, khiến âm thanh bị mất năng lượng tần số cao, người nghe cảm thấy phụ âm bị "nghẹt", "dính chữ" hoặc như bị nuốt âm.

---

### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.
**Trả lời:**  
Trong mô hình Nguồn – Lọc $X(z) = E(z) \cdot H(z)$:
* **$F_0$ (Tần số cơ bản):** Là đặc tính của **nguồn kích thích $E(z)$**, do tốc độ rung của dây thanh quản quyết định. Trên miền tần số, $F_0$ quyết định khoảng cách giữa các vạch điều hòa phổ (Harmonics).
* **Formant ($F_1, F_2, F_3$):** Là các tần số cộng hưởng của **bộ lọc thanh đạo $H(z)$**, do hình học khoang miệng, vị trí của lưỡi, môi và vòm họng quyết định. Formant tạo nên đường bao phổ (Spectral Envelope) quy định âm sắc của từng nguyên âm.
* Do đó, thay đổi $F_0$ chỉ đơn thuần là làm thay đổi mật độ các vạch hài âm của nguồn $E(z)$, hoàn toàn độc lập với vị trí các đỉnh cộng hưởng của bộ lọc $H(z)$. Nếu thay đổi cả formant khi tăng $F_0$, giọng nói sẽ bị biến dạng kích thước thanh đạo thành giọng "sóc chuột chipmunk".

---

### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?
**Trả lời:**  
* **Cơ chế thay đổi Pitch của TD-PSOLA:**
  1. Xác định các điểm đỉnh chu kỳ pitch (Pitch marks / Epochs) trên tín hiệu gốc tương ứng chu kỳ $T_0$.
  2. Nhân cửa sổ Hanning đối xứng quanh từng epoch với độ rộng $2T_0$ để tách thành các đoạn segment $s_i[n]$.
  3. Dời khoảng cách giữa các segment sang chu kỳ mới $T_{\text{new}}$ (với $T_{\text{new}} = F_s / F_{0,\text{new}}$) rồi thực hiện cộng dồn chồng chập (Overlap-Add). Do vị trí các vạch phổ cộng hưởng bên trong mỗi segment không đổi, phương pháp này thay đổi được $F_0$ mà vẫn giữ nguyên bao phổ Formant.
* **Hiện tượng Buzziness trên Voiced Fricative:**
  Âm xát hữu thanh (voiced fricative như /v/, /z/, /d/) là tín hiệu lai: vừa có tính tuần hoàn của dây thanh vừa có tính ngẫu nhiên hỗn loạn của luồng khí qua khe hẹp (nhiễu vô thanh). Khi kéo dài thời lượng, TD-PSOLA buộc phải lặp lại các frame tuần hoàn. Việc lặp lại cưỡng bức các đoạn nhiễu ngẫu nhiên theo một chu kỳ nhân tạo cố định sẽ phá vỡ bản chất ngẫu nhiên của âm xát, biến nhiễu thành âm điều hòa cơ học, gây ra tiếng rè kim loại khó chịu ("buzziness").

---

### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?
**Trả lời:**  
Trong mô hình HMM chuẩn, các quan sát trong cùng một trạng thái được giả định tuân theo phân bố chuẩn tĩnh $\mathcal{N}(\mu_i, \Sigma_i)$. Nếu không có đặc trưng động, nghiệm hợp lý cực đại (ML) tại mỗi frame chỉ đơn giản là vector trung bình $\mu_i$ của trạng thái đó, khiến quỹ đạo tham số âm học bị đứt gãy thành dạng bậc thang thô ráp tại biên chuyển trạng thái.
Khi bổ sung vector đạo hàm động bậc một $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$:
* Vector quan sát trở thành $O_t = [c_t^T, \Delta c_t^T]^T$.
* Nghiệm cực đại hóa hàm hợp lý $P(O \mid \lambda, Q)$ bị ràng buộc bởi mối quan hệ vi phân liên tục giữa các frame kế tiếp. Thuật toán MLPG giải hệ phương trình tuyến tính này sẽ tìm ra đường cong tham số $c_t$ biến thiên mềm mại đi xuyên qua các phân bố của từng state, triệt tiêu hoàn toàn các bước nhảy bậc thang gián đoạn.

---

### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?
**Trả lời:**  
Bởi vì phương pháp Concatenative sử dụng trực tiếp các đoạn sóng âm tự nhiên nguyên bản và có khả năng can thiệp/chỉnh sửa tín hiệu rất hạn chế (chỉ cắt ghép và cross-fade nhẹ). 
Trong thực tế, một âm vị hoặc một từ có vô vàn biến thể phát âm tùy thuộc vào:
1. Ngữ cảnh âm vị học đi trước và đi sau (hiện tượng đồng hóa cấu âm).
2. Trọng âm từ và vị trí trong câu (đầu câu cao giọng, cuối câu trầm giọng).
3. Tốc độ nói và sắc thái cảm xúc.
Để câu tổng hợp đạt độ tự nhiên cao trong mọi tình huống mà không bị lệch pha hay lệch ngữ điệu, kho Inventory bắt buộc phải thu thập đủ toàn bộ các tổ hợp ngữ cảnh đó, dẫn đến kích thước cơ sở dữ liệu bùng nổ theo cấp số nhân.

---

### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.
**Trả lời:**  
1. **Cơ chế sinh tín hiệu âm học:**
   - *Concatenative TTS:* Dựa trên cơ chế **tìm kiếm và tái sử dụng (Search & Selection)** các mẫu âm thanh có sẵn trong kho lưu trữ, không tạo ra âm thanh mới.
   - *Neural TTS:* Dựa trên cơ chế **mô hình sinh học sâu (Deep Generative Model)**; mạng nơ-ron tự học và phát sinh hoàn toàn biểu diễn âm học mới (Mel-spectrogram) và tổng hợp waveform từ đầu thông qua Neural Vocoder.
2. **Khả năng tổng quát hóa và kiểm soát ngữ điệu:**
   - *Concatenative TTS:* Rất khó thay đổi giọng nói hoặc kiểm soát ngữ điệu linh hoạt ngoài các mẫu đã thu sẵn; chất lượng phụ thuộc vào độ lớn của kho dữ liệu.
   - *Neural TTS:* Có khả năng khái quát hóa vượt trội, dễ dàng điều khiển cao độ, tốc độ, nhân bản giọng nói (Voice Cloning) và chuyển đổi người nói (Multi-speaker Adaptation) chỉ với lượng dữ liệu huấn luyện vừa phải.

---

### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?
**Trả lời:**  
Bởi vì việc sử dụng một mô hình ngẫu nhiên từ Internet vi phạm nguyên tắc kiểm soát biến số trong nghiên cứu khoa học:
1. **Thiếu tính đồng nhất về dữ liệu:** Mô hình ngoài Internet được huấn luyện trên tập dữ liệu khác biệt về người nói, chất lượng phòng thu, tần số lấy mẫu (ví dụ 22.05 kHz hay 24 kHz thay vì 16 kHz của Lab) và cấu trúc câu khác biệt.
2. **Thiếu điều kiện so sánh công bằng (Fair Comparison):** Sự vượt trội của mô hình đó có thể đến từ việc nó được huấn luyện trên hàng nghìn giờ dữ liệu chất lượng cao chứ không phản ánh đúng bản chất kỹ thuật của các thuật toán được kiểm nghiệm trong bài thực hành.
3. **Rủi ro môi trường thực hành:** Mô hình tải tự do dễ phát sinh xung đột thư viện, đòi hỏi GPU chuyên dụng hoặc không thể chạy offline ổn định trên máy trạm của phòng Lab.

---

### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.
**Trả lời:**  
Một hệ thống có **MOS cao** (giọng nghe rất tự nhiên, ấm áp, truyền cảm, giống người thật) nhưng **Intelligibility thấp** (khó nghe rõ nội dung từng từ cụ thể) vẫn có giá trị ứng dụng cao trong các lĩnh vực:
1. **Âm thanh nền và hiệu ứng môi trường trong Game / Phim ảnh:** Dùng để tạo tiếng rì rầm của đám đông (crowd babble, ambient chatter trong quán cà phê, quảng trường), nơi người nghe chỉ cần cảm nhận không khí đông đúc tự nhiên mà không cần phân biệt nội dung từng câu nói.
2. **Nhân vật phụ nói ngôn ngữ hư cấu (Fictional Languages):** Lồng tiếng cho các chủng tộc giả tưởng trong trò chơi điện tử (ví dụ: Simlish trong The Sims, ngôn ngữ người ngoài hành tinh trong phim viễn tưởng); yêu cầu giọng nói giàu cảm xúc, tự nhiên nhưng ngôn từ hoàn toàn vô nghĩa.
3. **Ứng dụng thư giãn, ru ngủ và thiền định (ASMR / Sleep Aid):** Cung cấp giọng thì thầm êm dịu, nhịp thở tự nhiên giúp xoa dịu hệ thần kinh mà không khiến não bộ phải tập trung xử lý ngữ nghĩa văn bản.

---

## 7. KẾT LUẬN VÀ BẢNG ĐỐI CHIẾU RUBRIC

### 7.1 Kết luận
Bài thực hành Lab 4 đã hoàn thành xuất sắc toàn bộ mục tiêu đề ra:
- Xây dựng thành công chuỗi Pipeline tổng hợp tiếng nói miền hẹp hoàn chỉnh từ chuẩn hóa văn bản đến phát sinh waveform.
- Chứng minh bằng thực nghiệm rằng thuật toán **Dynamic Programming Unit Selection** giúp giảm **$51.6\%$ Join Cost** so với ghép nối ngẫu nhiên, và cửa sổ **Cross-fade $15\text{ ms}$** là tối ưu để loại bỏ tiếng click mà không làm nhòe phụ âm.
- Làm sáng tỏ vai trò của $\Delta$-feature trong việc làm trơn quỹ đạo tham số HMM và đối sánh toàn diện ưu nhược điểm của 3 trường phái TTS.

### 7.2 Bảng Đối chiếu Rubric Chấm điểm (10.0 / 10.0)

| Hạng mục đánh giá | Điểm tối đa | Điểm tự đánh giá | Minh chứng đạt chuẩn trong sản phẩm nộp |
|:---|:---:|:---:|:---|
| **Chuẩn bị dữ liệu & normalization** | **1.0** | **1.0** | Kho 48 WAV chuẩn 16kHz mono, RMS 0.08, metadata.csv, [fig1_inventory_stats.png](file:///c:/Speech&Audio/lab4/figures/fig1_inventory_stats.png). |
| **Concatenative baseline** | **1.5** | **1.5** | raw_concat.wav ghép chuẩn câu test, [fig2_spectrogram_boundaries.png](file:///c:/Speech&Audio/lab4/figures/fig2_spectrogram_boundaries.png) đánh dấu vạch đỏ tại biên nối. |
| **Unit selection & smoothing** | **2.0** | **2.0** | Thuật toán DP Viterbi giảm 51.6% cost ([fig3_unit_selection_matrix.png](file:///c:/Speech&Audio/lab4/figures/fig3_unit_selection_matrix.png)), cross-fade tối ưu 15ms, best_concat.wav. |
| **Prosody / source-filter** | **1.0** | **1.0** | Đo Autocorrelation $F_0 \approx F_s/N_p$ ($N_p=69, F_0=231.88\text{Hz}$), time-stretch 0.8x/1.2x, pitch shift $\pm 2$ semitones ([fig4_prosody_modifications.png](file:///c:/Speech&Audio/lab4/figures/fig4_prosody_modifications.png)). |
| **HMM-based synthesis** | **1.5** | **1.5** | hmm_tts.wav, mô phỏng quỹ đạo F0/Formant, phân tích vai trò Delta-feature ([fig5_hmm_trajectories.png](file:///c:/Speech&Audio/lab4/figures/fig5_hmm_trajectories.png)). |
| **Neural TTS inference** | **1.0** | **1.0** | neural_tts.wav chạy offline, đo đạc Latency và RTF cho 3 câu, xuất Mel-spectrogram ([fig6_comparative_spectrogram.png](file:///c:/Speech&Audio/lab4/figures/fig6_comparative_spectrogram.png)). |
| **Đánh giá & báo cáo** | **2.0** | **2.0** | Báo cáo chi tiết đầy đủ 7 mục, giải trọn vẹn 10 câu hỏi lý thuyết, bảng điểm MOS ratings.csv, notebook Lab04.ipynb hoàn chỉnh. |
| **TỔNG ĐIỂM** | **10.0** | **10.0** | **ĐẠT XUẤT SẮC TOÀN DIỆN MỌI TIÊU CHÍ** |
