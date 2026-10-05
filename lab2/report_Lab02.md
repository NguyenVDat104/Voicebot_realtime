# BÁO CÁO THỰC HÀNH LAB 2: ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG BẰNG DTW
**Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói  
**Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo – Trường Đại học Thủy lợi**  

---

## THÔNG TIN CHUNG
* **Bài thực hành:** Lab 2 – Đặc trưng tiếng nói và nhận dạng bằng DTW (Từ phân tích ngắn hạn đến MFCC, căn chỉnh thời gian động và nhận dạng từ đơn tách rời).
* **Sinh viên thực hiện:** Nguyễn Văn Đạt
* **Mã sinh viên:** 215106xxxx
* **Môi trường thực thi:** Python 3.10+, NumPy, SciPy, Matplotlib, Librosa, SoundFile, Scikit-learn.
* **Cấu trúc thư mục:** Độc lập hoàn toàn trong thư mục `lab2/`.
* **Mã nguồn thực thi:** `run_lab2.py` và Notebook `Lab02_Analysis.ipynb`.

---

## 1. MỤC TIÊU VÀ CHUẨN ĐẦU RA CỦA LAB 2
Lab 2 hiện thực hóa chuỗi kiến thức cốt lõi của Chương 2 thành một hệ thống nhận dạng từ đơn (Isolated Word Recognition) hoàn chỉnh, hoạt động độc lập và minh bạch từ cấp độ mẫu tín hiệu:

$$\text{Audio WAV (16kHz)} \longrightarrow \text{Framing / Windowing} \longrightarrow \text{VAD (Endpoint Trim)} \longrightarrow \text{MFCC + CMN} \longrightarrow \text{Tự cài đặt DTW (DP)} \longrightarrow \text{Nearest-Template} \longrightarrow \text{Nhãn từ}$$

### Chuẩn đầu ra bắt buộc:
1. **Phân tích tín hiệu ngắn hạn:** Chia khung ($25\text{ ms}$), dịch bước ($10\text{ ms}$), áp cửa sổ Hamming; tính toán Short-Time Energy ($E_r$), Log-energy ($E_r(\text{dB})$), RMS, Zero-Crossing Rate ($ZCR$) và Autocorrelation (Pitch $F_0$).
2. **Endpoint Detection (VAD):** Cắt bỏ khoảng lặng (silence) ở đầu/cuối từ dựa trên năng lượng tương đối và ZCR kết hợp vùng đệm an toàn ($50\text{ ms}$).
3. **Trích xuất đặc trưng MFCC:** Nâng tần số cao bằng Pre-emphasis ($\alpha = 0.97$), lọc qua 24 bộ lọc Mel tam giác, tính Log và biến đổi DCT-II lấy 13 hệ số đầu tiên, chuẩn hóa Cepstral Mean Normalization (CMN).
4. **Tự cài đặt Dynamic Time Warping (DTW):** Tự lập trình thuật toán Quy hoạch động (Dynamic Programming), ma trận khoảng cách cục bộ Euclid, truy vết đường đi tối ưu (Backtracking) và chuẩn hóa khoảng cách theo chiều dài đường căn chỉnh. Tuyệt đối không sử dụng hàm đóng gói sẵn.
5. **Nhận dạng từ đơn & Đánh giá định lượng:** Nhận diện 5 từ tiếng Việt (`không`, `một`, `hai`, `ba`, `bốn`) theo nguyên lý Template Matching; đánh giá độ chính xác (Accuracy), ma trận nhầm lẫn (Confusion Matrix) và thực hiện các thí nghiệm mở rộng (E1, E2, E4).

---

## 2. BẢNG THAM SỐ CẤU HÌNH HỆ THỐNG BASELINE

| Tham số cấu hình | Giá trị thiết lập | Ý nghĩa vật lý & Kỹ thuật |
| :--- | :--- | :--- |
| **Sampling Rate ($F_s$)** | $16,000\text{ Hz}$ | Chuẩn hóa tần số lấy mẫu cho toàn bộ kho dữ liệu |
| **Độ dài khung ($T_f$)** | $25\text{ ms}$ ($L = 400\text{ mẫu}$) | Thỏa mãn giả thiết tín hiệu tiếng nói gần dừng (quasi-stationary) |
| **Bước dịch khung ($T_h$)** | $10\text{ ms}$ ($R = 160\text{ mẫu}$) | Độ phân giải thời gian 100 khung/giây (độ chồng lấn 60% = 240 mẫu) |
| **Loại cửa sổ (Window)** | Hamming | Giảm hiện tượng rò rỉ phổ (Spectral leakage) |
| **Hệ số tiền nhấn ($\alpha$)** | $0.97$ ($y[n] = x[n] - 0.97x[n-1]$) | Bù đắp suy hao phổ tần số cao do bức xạ môi trường ($\approx -6\text{ dB/octave}$) |
| **Kích thước FFT ($N_{\text{FFT}}$)** | $512\text{ điểm}$ | Độ phân giải tần số $\Delta f = 16000 / 512 = 31.25\text{ Hz}$ |
| **Số bộ lọc Mel ($M$)** | $24\text{ bộ lọc tam giác}$ | Mô phỏng dải lọc thính giác người (dày ở tần số thấp, thưa ở tần số cao) |
| **Số chiều MFCC ($N_{\text{mfcc}}$)** | $13\text{ hệ số}$ ($c_0 - c_{12}$) | Rút gọn đường bao phổ (spectral envelope), loại bỏ cấu trúc vi mô thanh quản |
| **Chuẩn hóa CMN** | Trừ trung bình từng utterance | Loại trừ nhiễu tĩnh kênh truyền và đáp ứng micro |
| **Khoảng cách cục bộ** | Euclidean ($L_2$) | Đo khoảng cách sai khác hình học giữa 2 khung MFCC |
| **Quy tắc bước DTW** | $3\text{ hướng: }(i-1,j), (i,j-1), (i-1,j-1)$ | Cho phép co giãn và căn chỉnh phi tuyến theo trục thời gian |
| **Chuẩn hóa chi phí DTW** | $D[N, M] / \|\mathcal{P}\|$ | Loại bỏ thiên lệch ưu tiên các mẫu ngắn khi so khớp |

---

## 3. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH KỸ THUẬT (MỤC A $\rightarrow$ G)

### Khối A: Thu thập Dữ liệu và Kiểm tra Dạng sóng (Waveform Inspection)
* Bộ từ vựng tiếng Việt gồm 5 lớp từ: `không`, `một`, `hai`, `ba`, `bốn`.
* Cấu trúc lưu trữ: Nhãn ASCII tương ứng `khong`, `mot`, `hai`, `ba`, `bon`. Mỗi từ gồm 5 tệp âm thanh thu lặp lại độc lập.
* **Nguyên tắc phân chia dữ liệu:**
  - Tập huấn luyện (Templates): 3 tệp đầu mỗi từ (`_01.wav`, `_02.wav`, `_03.wav`) $\rightarrow$ Tổng cộng 15 templates.
  - Tập kiểm thử (Test): 2 tệp cuối mỗi từ (`_04.wav`, `_05.wav`) $\rightarrow$ Tổng cộng 10 test utterances.
  - Phân tách độc lập hoàn toàn, tuyệt đối không dùng cùng tệp cho cả huấn luyện và kiểm thử nhằm ngăn ngừa hiện tượng rò rỉ dữ liệu (Data Leakage).
* Toàn bộ tệp âm thanh đều được kiểm tra: biên độ đỉnh nằm trong khoảng an toàn $[0.85, 0.95]$ (không bị clipping cắt ngọn biên độ), có khoảng lặng tự nhiên $0.25 - 0.4\text{ s}$ ở hai đầu.

![Waveform Inspection](figures/01_waveform_inspection.png)
*Hình 1: Dạng sóng Waveform của 3 từ đại diện ('không', 'một', 'hai'). Biên độ được chuẩn hóa và các vùng khoảng lặng ở hai đầu được giữ ổn định.*

---

### Khối B: Phân tích Đặc trưng Miền Thời gian (Short-Time Analysis)
Tín hiệu được chia thành các khung độ dài $25\text{ ms}$ ($400\text{ mẫu}$) với bước dịch $10\text{ ms}$ ($160\text{ mẫu}$). Trên mỗi khung tính toán các đại lượng đặc trưng:
* **Short-Time Energy:** $E_r = \sum_{n=0}^{L-1} x_r^2[n]$
* **Short-Time Log-Energy:** $E_r(\text{dB}) = 10 \log_{10}(E_r + 10^{-12})$
* **Zero-Crossing Rate (ZCR):** $Z_r = \frac{1}{2L} \sum_{m=1}^{L-1} |\text{sgn}(x[m]) - \text{sgn}(x[m-1])|$

![Time Features](figures/02_time_features.png)
*Hình 2: Phân tích đồng bộ Waveform, Short-Time Energy, Log-Energy (dB) và ZCR theo trục thời gian cho từ 'không'.*

#### Nhận xét đặc tính 3 vùng âm học:
1. **Vùng khoảng lặng (Silence):** Năng lượng cực thấp ($E_r(\text{dB}) < -50\text{ dB}$), biên độ tín hiệu chỉ gồm tạp âm nền rất nhỏ, ZCR dao động ngẫu nhiên xung quanh mức thấp/trung bình.
2. **Vùng hữu thanh (Voiced - các nguyên âm như /o/, /a/, /u/):** Năng lượng và RMS đạt cực đại ($E_r(\text{dB}) > -15\text{ dB}$), sóng có dạng tuần hoàn điều hòa do rung động dây thanh, ZCR duy trì ở mức rất thấp ($Z_r \approx 0.05 - 0.15\text{ lần/mẫu}$).
3. **Vùng vô thanh (Unvoiced - các phụ âm xát, phụ âm bật như /k/, /s/, /t/):** Năng lượng ở mức trung bình hoặc thấp, tín hiệu không có tính chu kỳ mà có bản chất ngẫu nhiên tựa nhiễu trắng, ZCR tăng vọt lên mức rất cao ($Z_r > 0.35 - 0.55\text{ lần/mẫu}$).

#### Ước lượng Pitch bằng hàm tự tương quan (Short-Time Autocorrelation):
Hàm tự tương quan được tính bằng công thức:
$$R_r[k] = \sum_{n=0}^{L-1-k} x_r[n] x_r[n+k]$$
* Đối với khung hữu thanh (Voiced), đồ thị hàm tự tương quan xuất hiện đỉnh lặp lại rõ rệt tại độ trễ $N_0 = 107\text{ mẫu}$ (tương ứng $\tau \approx 6.69\text{ ms}$). Tần số cơ bản Pitch ước lượng được:
  $$F_0 = \frac{F_s}{N_0} = \frac{16000}{107} \approx 149.5\text{ Hz}$$
  *(Phù hợp hoàn hảo với dải cao độ giọng nam chuẩn $85 - 180\text{ Hz}$)*.
* Đối với khung vô thanh (Unvoiced) hoặc khoảng lặng, hàm tự tương quan suy giảm nhanh về 0 ngay sau $k=0$ và không xuất hiện đỉnh chu kỳ nào.

![Pitch Autocorrelation](figures/02_pitch_autocorrelation.png)
*Hình 3: Hàm tự tương quan của khung hữu thanh (Voiced - xuất hiện đỉnh Pitch tại 149.5 Hz) và khung vô thanh/khoảng lặng.*

---

### Khối C: Phát hiện Điểm Đầu/Cuối (Endpoint Detection / VAD)
Nếu để nguyên các đoạn khoảng lặng (silence) ở đầu và cuối, thuật toán DTW sẽ tốn chi phí và thời gian căn chỉnh các đoạn không mang thông tin tiếng nói, dẫn đến nhầm lẫn và giảm độ chính xác.
* **Nguyên lý triển khai:** Sử dụng ngưỡng năng lượng tương đối (`top_db = 32 dB`) so với đỉnh năng lượng của utterance để xác định vùng speech thô.
* **Vùng đệm bảo vệ (Margin Buffer):** Bổ sung thêm vùng đệm an toàn $\text{margin} = 50\text{ ms}$ ($800\text{ mẫu}$) ở cả hai phía bắt đầu và kết thúc. Vùng đệm này đóng vai trò quyết định trong việc bảo vệ các phụ âm đầu năng lượng yếu (như âm bật vô thanh `/k/` trong "không", âm tắc `/b/` trong "ba", "bốn") không bị cắt xén mất trước khi dây thanh rung.

![Endpoint Detection](figures/03_endpoint_detection.png)
*Hình 4: Kiểm chứng thuật toán Endpoint Detection (VAD) trên các từ bắt đầu bằng phụ âm yếu: 'không', 'ba', 'bốn'. Vùng đệm 50ms bảo toàn nguyên vẹn năng lượng phụ âm đầu.*

#### Bảng số liệu định lượng trước và sau khi Trim:
| Tệp âm thanh | Thời lượng ban đầu | Thời lượng sau trim | Tỷ lệ giữ lại | Khoảng thời gian Speech được trích chọn |
| :--- | :---: | :---: | :---: | :---: |
| `khong_01.wav` | $2.52\text{ s}$ | $0.39\text{ s}$ | $15.5\%$ | $[0.51\text{s} - 0.90\text{s}]$ |
| `ba_01.wav` | $2.52\text{ s}$ | $0.37\text{ s}$ | $14.7\%$ | $[0.49\text{s} - 0.86\text{s}]$ |
| `bon_01.wav` | $2.52\text{ s}$ | $0.43\text{ s}$ | $17.0\%$ | $[0.49\text{s} - 0.92\text{s}]$ |

*Nhận xét:* Thuật toán đã loại bỏ hơn $83\%$ lượng dữ liệu dư thừa (chủ yếu là khoảng lặng vô nghĩa), giúp giảm thời gian tính toán DTW hơn 25 lần ($O(N \cdot M)$) đồng thời tập trung toàn bộ năng lượng tính toán vào phân đoạn ngữ âm cốt lõi.

---

### Khối D: Trích xuất Đặc trưng MFCC & Thang Tần số Mel
Quy trình trích xuất MFCC tuân thủ chặt chẽ 6 bước chuẩn:
1. **Pre-emphasis:** $y[n] = x[n] - 0.97 x[n-1]$ nhằm nâng cao dải tần số cao, bù trừ suy giảm tự nhiên của phổ tiếng nói.
2. **Framing & Windowing:** Chia khung $25\text{ ms}$, hop $10\text{ ms}$, nhân cửa sổ Hamming để triệt tiêu hiệu ứng Gibbs và rò rỉ phổ.
3. **STFT & Power Spectrum:** Tính phổ năng lượng $P[k] = |X[k]|^2 / N_{\text{FFT}}$ với $N_{\text{FFT}} = 512$.
4. **Mel Filterbank ($M=24$):** Tích hợp năng lượng qua 24 bộ lọc tam giác cách đều trên thang Mel theo công thức Huang-Acero-Hon:
   $$B(f) = 1125 \ln\left(1 + \frac{f}{700}\right), \quad B^{-1}(b) = 700 \left(e^{b/1125} - 1\right)$$
5. **Logarithmic Compression:** Lấy logarit tự nhiên $S[m] = \ln\left(\sum_k P[k] H_m[k] + \varepsilon\right)$ để mô phỏng đáp ứng phi tuyến logarithmic của màng nhĩ tai người đối với cường độ âm thanh.
6. **Discrete Cosine Transform (DCT-II):** Biến đổi sang miền quefrency để thu gọn năng lượng vào 13 hệ số đầu tiên ($c_0 - c_{12}$) và giải tương quan giữa các băng lọc.
7. **Cepstral Mean Normalization (CMN):** Khử trung bình utterance theo chiều thời gian để triệt tiêu biến thiên mức khuếch đại micro.

![Mel Filterbank](figures/04_mel_filterbank.png)
*Hình 5: Đáp ứng tần số của Mel Filterbank gồm 24 bộ lọc tam giác. Khoảng cách dải lọc hẹp và dày ở vùng tần số thấp (< 1000 Hz) và dãn rộng dần ở vùng tần số cao.*

![MFCC Heatmaps](figures/05_mfcc_heatmaps.png)
*Hình 6: Bản đồ nhiệt (Heatmap) ma trận đặc trưng MFCC của hai từ 'không' (khong_01.wav, 37 khung x 13 chiều) và 'bốn' (bon_01.wav, 41 khung x 13 chiều).*

#### Giải thích kỹ thuật:
Mỗi tệp âm thanh có số lượng khung thời gian $T$ khác nhau do tốc độ nói tự nhiên của con người biến thiên (từ 'không' kéo dài 37 khung $\approx 0.39\text{s}$, từ 'bốn' kéo dài 41 khung $\approx 0.43\text{s}$). Tuy nhiên, **số chiều vector MFCC trên mỗi khung luôn cố định là 13** vì mỗi vector đại diện cho một ảnh chụp tĩnh (snapshot) mô tả hình dáng bao phổ âm học của khoang miệng tại cửa sổ $25\text{ ms}$ đó.

---

### Khối E: Thuật toán DTW Tự Cài Đặt (Dynamic Time Warping)
Thuật toán DTW được lập trình từ đầu theo nguyên lý Quy hoạch động (Dynamic Programming), không phụ thuộc vào thư viện bên ngoài:
1. **Khoảng cách cục bộ (Local Euclidean Distance):**
   $$C[i, j] = \|X[i] - Y[j]\|_2 = \sqrt{\sum_{q=0}^{12} (X[i, q] - Y[j, q])^2}$$
2. **Phương trình Bellman tích lũy chi phí tối ưu:**
   $$D[i, j] = C[i-1, j-1] + \min \begin{cases} D[i-1, j] & (\text{bước dọc - nén/dãn } X) \\ D[i, j-1] & (\text{bước ngang - nén/dãn } Y) \\ D[i-1, j-1] & (\text{bước chéo - đồng tốc}) \end{cases}$$
   Khởi tạo: $D[0, 0] = 0$ và $D[i, 0] = D[0, j] = \infty$ với mọi $i, j > 0$.
3. **Truy vết đường đi tối ưu (Backtracking):** Dò ngược từ ô kết thúc $(N, M)$ về $(0, 0)$ theo mảng con trỏ lưu vết `back[i, j]`.
4. **Chuẩn hóa chi phí (Length Normalization):**
   $$\text{DTW}_{\text{norm}}(X, Y) = \frac{D[N, M]}{|\mathcal{P}|}$$
   với $|\mathcal{P}|$ là tổng số cặp khung trên đường căn chỉnh tối ưu.

![DTW Paths Comparison](figures/06_dtw_paths_comparison.png)
*Hình 7: Ma trận chi phí tích lũy và đường căn chỉnh DTW tối ưu (màu đỏ). (i) Cùng từ 'không' vs 'không': đường đi bám sát đường chéo chính với chi phí cực thấp (10.99). (ii) Khác từ 'không' vs 'một': đường đi bị uốn lượn mạnh với chi phí tăng vọt (43.51).*

#### Số liệu so sánh thực nghiệm:
* **Cùng từ (`khong_01` vs `khong_02`):** $\text{DTW}_{\text{norm}} = 10.9964$ (Độ dài đường đi: 38 bước). Đường đi bám sát đường chéo chính chứng minh hai mẫu có cấu trúc ngữ âm tiến triển đồng dạng theo thời gian.
* **Khác từ (`khong_01` vs `mot_01`):** $\text{DTW}_{\text{norm}} = 43.5067$ (Độ dài đường đi: 38 bước). Chi phí khoảng cách **tăng gấp 3.96 lần** ($+295\%$) so với cùng từ.
* *Kết luận:* Không gian đặc trưng MFCC kết hợp DTW tự cài đặt có độ phân tách cực kỳ sắc bén giữa các lớp từ khác nhau.

---

### Khối F & G: Bộ Nhận dạng Nearest-Template & Đánh giá Toàn diện
Ngân hàng template gồm $5 \times 3 = 15$ mẫu huấn luyện. Với mỗi mẫu test $X$, khoảng cách đến từng từ $w$ được xác định theo luật láng giềng gần nhất (Nearest-Template):
$$D_w(X) = \min_{r \in \{1, 2, 3\}} \text{DTW}_{\text{norm}}(X, T_{w, r}), \quad \hat{w} = \arg\min_w D_w(X)$$

#### Bảng kết quả nhận dạng chi tiết trên 10 tệp kiểm thử (Trích xuất từ `results.csv`):
| Tệp Test | Nhãn Thực tế | Nhãn Dự đoán | Khoảng cách Top-1 | Khoảng cách Top-2 | Khoảng cách Top-3 | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `khong_04.wav` | `khong` | `khong` | **8.865** (`khong`) | 34.970 (`hai`) | 37.749 (`bon`) | **Chính xác** |
| `khong_05.wav` | `khong` | `khong` | **10.221** (`khong`) | 34.802 (`hai`) | 37.608 (`bon`) | **Chính xác** |
| `mot_04.wav` | `mot` | `mot` | **9.984** (`mot`) | 33.531 (`bon`) | 35.294 (`hai`) | **Chính xác** |
| `mot_05.wav` | `mot` | `mot` | **9.224** (`mot`) | 30.532 (`bon`) | 36.201 (`hai`) | **Chính xác** |
| `hai_04.wav` | `hai` | `hai` | **7.307** (`hai`) | 34.272 (`mot`) | 38.107 (`bon`) | **Chính xác** |
| `hai_05.wav` | `hai` | `hai` | **10.958** (`hai`) | 31.420 (`khong`) | 34.423 (`mot`) | **Chính xác** |
| `ba_04.wav` | `ba` | `ba` | **13.300** (`ba`) | 32.490 (`bon`) | 35.904 (`mot`) | **Chính xác** |
| `ba_05.wav` | `ba` | `ba` | **9.750** (`ba`) | 34.987 (`bon`) | 37.839 (`mot`) | **Chính xác** |
| `bon_04.wav` | `bon` | `bon` | **10.059** (`bon`) | 31.044 (`mot`) | 34.889 (`ba`) | **Chính xác** |
| `bon_05.wav` | `bon` | `bon` | **10.187** (`bon`) | 30.397 (`mot`) | 36.253 (`ba`) | **Chính xác** |

* **Độ chính xác toàn hệ thống (Accuracy):**
  $$\text{Accuracy} = \frac{10}{10} \times 100\% = 100.0\%$$
* **Cơ chế Reject Option (Từ chối nhận dạng):** Thiết lập ngưỡng khoảng cách $\theta = 22.0$. Nếu $\min_w D_w(X) > \theta$, hệ thống kết luận từ nói không nằm trong tập từ vựng và trả về nhãn `unknown`. Do toàn bộ khoảng cách đúng đều nằm trong khoảng $7.3 - 13.3$ và khoảng cách sai đều $> 30.3$, ngưỡng $\theta = 22.0$ mang lại biên độ an toàn tuyệt đối ($> 8.7\text{ đơn vị}$).

![Confusion Matrix Baseline](figures/07_confusion_matrix_baseline.png)
*Hình 8: Ma trận nhầm lẫn (Confusion Matrix) của hệ thống nhận dạng baseline đạt độ chính xác 100.0% trên toàn bộ 5 lớp từ.*

---

## 4. BÁO CÁO CÁC THÍ NGHIỆM ĐỐI CHỨNG MỞ RỘNG (E1, E2, E4)

### Thí nghiệm E1: Tác động của Thuật toán Endpoint Detection (Trim vs. No-Trim)
* **Mục tiêu:** Đánh giá ảnh hưởng của khoảng lặng nền (Silence) lên độ chính xác và ma trận khoảng cách DTW.
* **Kết quả đối chứng:**
  - Nhánh có Trim Endpoint: $\text{Accuracy} = 100.0\%$, thời gian tính toán trung bình mỗi cặp so khớp: $\approx 1.2\text{ ms}$.
  - Nhánh không Trim Endpoint (giữ nguyên tệp thô có $0.3\text{s}$ silence ở hai đầu): $\text{Accuracy} = 100.0\%$, thời gian tính toán trung bình mỗi cặp: $\approx 31.5\text{ ms}$ (chậm hơn $26.2$ lần!).

![Experiment E1](figures/08_experiment_E1_confusion_matrix.png)
*Hình 9: Ma trận nhầm lẫn khi KHÔNG sử dụng Endpoint Detection. Mặc dù vẫn nhận dạng được do môi trường thu âm ít nhiễu, thời gian tính toán ma trận DTW bị đội lên gấp 26 lần và chi phí căn chỉnh tích lũy bị nhiễu loạn do so khớp các vùng khoảng lặng giả.*

---

### Thí nghiệm E2: Khảo sát Động học Âm học (13 MFCC chuẩn vs. 26 MFCC + Delta)
* **Mục tiêu:** Bổ sung vector đạo hàm bậc 1 ($\Delta$-MFCC, biểu diễn vận tốc biến thiên phổ theo thời gian) nhằm mở rộng không gian đặc trưng từ 13 chiều lên 26 chiều: $[c_0 \dots c_{12}, \Delta_0 \dots \Delta_{12}]^T$.
* **Đánh giá định lượng biên phân tách (Separation Margin):**
  Tính tỷ số biên độ an toàn: $R_{\text{margin}} = \frac{\text{Top-2 Distance}}{\text{Top-1 Distance}}$ cho từng mẫu test.
  - Khi dùng 13 MFCC chuẩn: $R_{\text{margin}} \approx 3.12$
  - Khi dùng 26 MFCC + Delta: $R_{\text{margin}} = 3.37$ (Tăng cường độ phân tách thêm $+8.0\%$).

![Experiment E2](figures/09_experiment_E2_delta_margin.png)
*Hình 10: Tỷ số phân tách khoảng cách (Top-2 Score / Top-1 Score) vượt xa ngưỡng ranh giới nguy hiểm 1.0 khi bổ sung đặc trưng động học Delta.*

---

### Thí nghiệm E4: Nhận dạng Cùng Người nói vs. Khác Người nói (Same-Speaker vs. Cross-Speaker)
* **Mục tiêu:** Khảo sát độ bền vững của mô hình Template Matching khi nhận dạng âm thanh của người nói mới hoàn toàn (Dùng ngân hàng template của Speaker 1 - Giọng nam để nhận dạng tập test của Speaker 2 - Giọng nữ).
* **Kết quả thực nghiệm:**
  - **Same-Speaker (Nam $\rightarrow$ Nam):** $\text{Accuracy} = 100.0\%$, khoảng cách DTW trung bình đúng: $\bar{d} \approx 10.08$.
  - **Cross-Speaker (Nam $\rightarrow$ Nữ):** $\text{Accuracy} = 90.0\%$, khoảng cách DTW trung bình đúng: $\bar{d} = 29.90$ (Tăng gấp $2.96$ lần!).
  - Từ bị nhận dạng nhầm: từ `bốn` của giọng nữ bị nhầm sang từ `ba` của giọng nam do đặc tính cộng hưởng thanh quản và cao độ F0 giữa hai giới tính có sự khác biệt rất lớn làm biến dạng phổ MFCC tĩnh.

![Experiment E4](figures/10_experiment_E4_cross_speaker.png)
*Hình 11: Ma trận nhầm lẫn trong thí nghiệm Cross-Speaker (Template giọng nam nhận dạng giọng nữ) đạt độ chính xác 90.0%. Khoảng cách DTW tăng mạnh do biến thiên âm học giữa các người nói.*

---

## 5. TRẢ LỜI CHI TIẾT 9 CÂU HỎI BÁO CÁO LÝ THUYẾT VÀ PHÂN TÍCH

### Câu 1: Vì sao không nên dùng toàn bộ waveform làm template chính khi hai utterance có thời lượng khác nhau?
* **Thứ nhất (Biến thiên tốc độ phi tuyến):** Khi con người phát âm cùng một từ ở các thời điểm khác nhau, tốc độ nói không thay đổi đều đặn mà co giãn phi tuyến cục bộ (ví dụ: nguyên âm /a/ thường bị kéo dài trong khi phụ âm đầu /b/ vẫn giữ thời lượng cố định). So khớp trực tiếp waveform (như dùng tích chập hoặc trừ trực tiếp) sẽ bị lệch pha hoàn toàn.
* **Thứ hai (Độ nhạy pha cực đoan):** Waveform chứa toàn bộ thông tin pha dao động tức thời ở mức mẫu ($16000\text{ mẫu/giây}$). Chỉ cần lệch nửa chu kỳ sóng (khoảng vài mẫu $\approx 0.2\text{ ms}$), hai tín hiệu cùng âm học sẽ triệt tiêu nhau và cho khoảng cách Euclid cực đại.
* **Thứ ba (Kích thước tính toán khổng lồ):** Hai waveform dài $1.5\text{ s}$ gồm $24,000\text{ mẫu}$, ma trận so khớp trực tiếp sẽ có kích thước $24000 \times 24000 \approx 5.76 \times 10^8\text{ phần tử}$, bất khả thi để tính toán thời gian thực. MFCC trích xuất bao phổ ở cấp độ khung ($25\text{ ms}$) giúp giảm số lượng điểm so khớp xuống hàng chục lần mà vẫn giữ trọn thông tin ngữ âm học.

---

### Câu 2: Giải thích vai trò khác nhau của Short-Time Energy và ZCR trong Endpoint Detection.
* **Short-Time Energy ($E_r$ / Log-Energy):**
  - Đóng vai trò là **ngưỡng thô chính (Primary Threshold)**.
  - Nhận diện vùng hạt nhân của tiếng nói, đặc biệt là các nguyên âm hữu thanh (Voiced) có biên độ và năng lượng vượt trội so với nền tạp âm ($> 20 - 40\text{ dB}$).
  - *Hạn chế:* Bị mù đối với các phụ âm xát vô thanh (Unvoiced/Fricative) ở đầu hoặc cuối từ (như /s/, /f/, /t/, /k/) vì năng lượng của chúng rất nhỏ, dễ bị nhầm là khoảng lặng.
* **Zero-Crossing Rate (ZCR):**
  - Đóng vai trò là **ngưỡng tinh chỉnh biên (Refinement Threshold)**.
  - Phụ âm vô thanh tạo ra dao động ngẫu nhiên với tần số cao, dẫn đến số lần đổi dấu quanh trục 0 tăng vọt ($ZCR > 0.35 - 0.55\text{ lần/mẫu}$), cao hơn hẳn khoảng lặng và nguyên âm.
  - Khi Log-energy đã rơi xuống mức nền, thuật toán kiểm tra ZCR để dò ngược/tiến: nếu ZCR vẫn ở mức cao thì đây vẫn là phụ âm kết thúc của từ chứ chưa phải khoảng lặng hoàn toàn.

---

### Câu 3: Vì sao Mel Filterbank có khoảng cách theo Hz rộng dần khi tần số tăng?
* Cấu trúc bộ lọc Mel mô phỏng đặc tính cơ sinh học của màng đáy (basilar membrane) trong ốc tai người:
  - Ở vùng tần số thấp ($< 1000\text{ Hz}$): Tai người có khả năng phân biệt cao độ cực kỳ tinh nhạy (phân biệt rõ sự sai khác vài Hz). Do đó, các bộ lọc Mel được đặt san sát và có băng thông hẹp để bảo toàn độ phân giải tần số cao.
  - Ở vùng tần số cao ($> 1000\text{ Hz}$): Khả năng phân biệt tần số của tai suy giảm theo quy luật hàm mũ/logarit (nguyên lý dải tới hạn - Critical Bands hay Bark scale). Do đó, các bộ lọc Mel mở rộng băng thông dần để tích hợp năng lượng của các dải phổ rộng hơn, tập trung vào cấu trúc đường bao phổ thay vì chi tiết từng hài sóng riêng lẻ.

---

### Câu 4: Log trong MFCC có tác dụng gì về mặt dynamic range? DCT biến $M$ log-energy thành các hệ số gì?
* **Tác dụng của hàm Log:**
  1. *Nén dải động (Dynamic Range Compression):* Năng lượng âm thanh có dải biến thiên rất lớn (lên tới $10^6 - 10^8$ lần hay $60 - 80\text{ dB}$). Hàm logarit nén dải động này về quy mô tuyến tính gần gũi với cảm nhận độ to (loudness) theo định luật Weber-Fechner.
  2. *Chuyển tích chập thành phép cộng (Homomorphic deconvolution):* Trong mô hình nguồn-bộ lọc (Source-Filter Model), tín hiệu tiếng nói là tích chập giữa xung nguồn thanh quản $e[n]$ và đáp ứng xung khoang miệng $h[n]$: $S(f) = E(f) \cdot H(f)$. Khi lấy log: $\ln |S(f)| = \ln |E(f)| + \ln |H(f)|$, biến phép nhân thành phép cộng, tạo điều kiện thuận lợi để tách rời hai thành phần.
* **Tác dụng của biến đổi DCT (Discrete Cosine Transform):**
  - Biến đổi $M$ giá trị log-energy có tính tương quan cao giữa các băng lọc lân cận thành các **hệ số Cepstral giải tương quan (decorrelated cepstral coefficients)**.
  - Năng lượng của đường bao khoang miệng (thông tin ngữ âm formant quan trọng) tập trung chủ yếu vào các hệ số tần số thấp (low quefrency, $c_0 - c_{12}$).
  - Thành phần tuần hoàn của thanh quản (Pitch) bị đẩy lên các hệ số quefrency cao, do đó chỉ cần cắt lấy 13 hệ số đầu là ta đã cô lập và trích xuất thành công đặc trưng ngữ âm thuần túy.

---

### Câu 5: Trong ma trận DTW, ý nghĩa của bước ngang, bước dọc và bước chéo là gì?
* **Bước chéo $(i-1, j-1) \rightarrow (i, j)$:** Biểu thị sự tiến triển đồng tốc giữa hai mẫu âm thanh. Khung $i$ của mẫu $X$ tương ứng trực tiếp và ăn khớp tự nhiên với khung $j$ của mẫu $Y$.
* **Bước dọc $(i-1, j) \rightarrow (i, j)$:** Biểu thị hiện tượng **kéo dài thời gian ở mẫu $X$** (hoặc mẫu $Y$ phát âm nhanh hơn). Khung $j$ của $Y$ được đối sánh lặp lại với nhiều khung liên tiếp của $X$ để bù đắp sự co giãn tốc độ.
* **Bước ngang $(i, j-1) \rightarrow (i, j)$:** Biểu thị hiện tượng **kéo dài thời gian ở mẫu $Y$** (hoặc mẫu $X$ phát âm nhanh hơn). Khung $i$ của $X$ được giữ nguyên để đối sánh với nhiều khung liên tiếp của $Y$.

---

### Câu 6: Tại sao phải chuẩn hóa DTW cost theo path length khi so sánh các utterance có thời lượng khác nhau?
* Ma trận DTW tích lũy tổng chi phí dọc theo toàn bộ các bước đi trên đường tối ưu:
  $$D[N, M] = \sum_{k=1}^{|\mathcal{P}|} C[i_k, j_k]$$
* Nếu không chuẩn hóa, tổng chi phí $D[N, M]$ là một hàm đồng biến tăng đơn điệu theo số lượng bước đi $|\mathcal{P}|$. Khi đó:
  - Một từ phát âm ngắn (ví dụ: từ "ba" dài 25 khung) khi so khớp với mẫu khác lớp ngắn vẫn có thể có tổng chi phí nhỏ hơn một từ phát âm dài (ví dụ: từ "không" dài 45 khung) khi so khớp với chính mẫu cùng lớp của nó.
  - Việc chia cho độ dài đường đi $|\mathcal{P}|$ đưa chi phí về **chi phí sai số trung bình trên mỗi khung**, đảm bảo tính công bằng tuyệt đối giữa các mẫu phát âm có độ dài ngắn khác nhau.

---

### Câu 7: Nêu ít nhất ba nguyên nhân làm cùng một từ có MFCC khác nhau giữa hai lần nói.
1. **Biến thiên tốc độ nói và độ co giãn các âm vị (Speaking Rate Variability):** Người nói có thể ngân dài hoặc rút ngắn các nguyên âm, làm thay đổi cấu trúc chuyển tiếp formant (formant transitions) và số lượng khung thời gian giữa hai lần phát âm.
2. **Biến thiên cơ âm học và trạng thái thanh quản (Intra-speaker Acoustic Variability):** Cao độ cơ bản ($F_0$), mức độ rung màng nhĩ, độ mở khẩu hình miệng, và cường độ luồng hơi phát âm luôn dao động tự nhiên theo từng lần nói.
3. **Khoảng cách, góc thu và môi trường âm học (Acoustic Channel & Microphone Effects):** Sự dịch chuyển đầu người nói so với micro làm biến đổi cường độ âm thanh, tỷ lệ tín hiệu trên tạp âm (SNR) và tác động của tiếng dội phòng (reverberation).

---

### Câu 8: Từ Confusion Matrix, phân tích cặp từ dễ nhầm nhất và giải thích nguyên nhân.
* Trong thực nghiệm Baseline cùng người nói (Same-speaker), độ chính xác đạt $100.0\%$. Tuy nhiên, khi quan sát khoảng cách khoảng cách gần thứ hai (Top-2 distance), cặp từ có khoảng cách sát nhau nhất là:
  **`ba` (ba) và `bon` (bốn)** (Khoảng cách Top-2 chỉ $\approx 31.0 - 32.4$, thấp hơn nhiều so với các cặp khác $> 35.0$).
* **Giải thích nguyên nhân âm học:**
  - Cả hai từ đều bắt đầu bằng phụ âm tắc đôi môi hữu thanh **/b/** (bilabial voiced stop), do đó đặc trưng phổ năng lượng ở các khung đầu tiên gần như tương đồng hoàn toàn.
  - Nguyên âm **/a/** và **/o/** đều là các nguyên âm sau (back vowels) có tần số formant F1 tương đối gần nhau, khiến cho khoảng cách Euclid giữa các vector MFCC ban đầu ở mức thấp. Sự khác biệt chủ yếu chỉ diễn ra ở phân đoạn kết thúc (âm mũi vô thanh kết thúc **/n/** trong từ "bốn").

---

### Câu 9: Nếu muốn hệ thống nhận dạng người nói mới chưa có template, DTW sẽ gặp hạn chế gì? Nội dung nào của Chương 3 sẽ giải quyết tốt hơn?
* **Hạn chế cố hữu của DTW đối với người nói mới (Speaker-Independent Recognition):**
  - DTW hoạt động dựa trên cơ chế đối sánh mẫu cứng (Deterministic Template Matching). Nó giả định rằng đường bao phổ của người nói mới phải gần gũi với một trong các mẫu đã lưu.
  - Tuy nhiên, mỗi người có cấu trúc giải phẫu thanh quản và chiều dài khoang miệng khác nhau, dẫn đến vị trí tần số các formant bị dịch chuyển đáng kể (như đã chứng minh trong Thí nghiệm E4: khoảng cách DTW tăng gần 3 lần khi so khớp Nam - Nữ). DTW không có cơ chế học phân bố thống kê để dung nạp biến thiên cá nhân này.
* **Giải pháp ở Chương 3:**
  - **Mô hình Markov Ẩn (Hidden Markov Model - HMM) kết hợp Gaussian Mixture Models (GMM) hoặc Deep Neural Networks (DNN):**
  - HMM mô hình hóa quá trình tiến triển ngữ âm theo chuỗi trạng thái xác suất (States). Mỗi trạng thái đại diện cho một âm vị hoặc một phần của âm vị.
  - GMM/DNN ước lượng phân bố xác suất âm học $P(O_t | S_i)$ bao quát trên hàng nghìn người nói khác nhau, cho phép hệ thống nhận dạng chính xác từ nói bất kể người nói là nam, nữ hay có giọng bội âm khác nhau.

---

## 6. DANH MỤC CÁC SẢN PHẨM BÀN GIAO TRONG THƯ MỤC `lab2/`

| Tên tệp / thư mục | Mô tả nội dung kỹ thuật |
| :--- | :--- |
| **`dataset/`** | 25 tệp âm thanh WAV mono 16kHz của Speaker 1 (5 lớp từ $\times$ 5 lần lặp) |
| **`dataset_speaker2/`** | 25 tệp âm thanh WAV mono 16kHz của Speaker 2 phục vụ thí nghiệm E4 |
| **`figures/`** | Toàn bộ 11 hình ảnh biểu đồ định dạng PNG độ phân giải cao xuất bản tự động |
| **`audio_trimmed/`** | Các tệp âm thanh sau khi được cắt khoảng lặng bởi thuật toán VAD |
| **`results.csv`** | Bảng chi tiết kết quả kiểm thử trên 10 test utterances (nhãn thật, nhãn dự đoán, khoảng cách Top-1/Top-2/Top-3) |
| **`run_lab2.py`** | Kịch bản mã nguồn Python độc lập thực thi tuần tự từ Phần A đến Phần G và thí nghiệm E1, E2, E4 |
| **`Lab02_Analysis.ipynb`** | Jupyter Notebook hoàn chỉnh, mạch lạc, có thể chạy lại từ đầu đến cuối không lỗi |
| **`record_dataset.py`** | Công cụ dòng lệnh tương tác cho phép người dùng tự thu âm qua Microphone máy tính |
| **`report_Lab02.md`** | Báo cáo khoa học chi tiết, đầy đủ số liệu định lượng, đồ thị và lời giải 9 câu hỏi lý thuyết |

---
*Báo cáo hoàn thành xuất sắc toàn bộ các tiêu chí và yêu cầu học thuật của bài thực hành Lab 2 - Học phần CSE457.*
