FROM python:3.12-slim

# Cài Java (cho VnCoreNLP), và wget để tải model
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app và set làm thư mục làm việc
WORKDIR /app

COPY requirements.txt .

# Cài thư viện Python
RUN pip install -r requirements.txt

# Cài PyTorch (CPU-only)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy toàn bộ source vào trong image
COPY . .

# Tải model (sẽ không bị lặp nếu model đã tồn tại từ lần trước)
RUN python main.py download_only

# Tải model PhoBERT
# RUN python utils/download_phobert_model.py

# Mở cổng mặc định FastAPI
EXPOSE 8000

# Lệnh chạy app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
