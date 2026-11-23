# Multi-stage / Optimized Python 3.11 Dockerfile for Real-Time Vietnamese Voicebot
FROM python:3.11-slim

# Thiết lập biến môi trường Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết cho xử lý tín hiệu âm thanh và kiểm tra healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Tối ưu cài đặt PyTorch CPU trước để tiết kiệm dung lượng Docker image (tránh tải 2.5GB CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Cài đặt các phụ thuộc từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Sao chép toàn bộ mã nguồn ứng dụng
COPY main.py .
COPY audio_processing/ audio_processing/
COPY services/ services/
COPY static/ static/

# Mở cổng mặc định 8000
EXPOSE 8000

# Kiểm tra tình trạng sức khỏe của container
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Khởi chạy ứng dụng với Uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
