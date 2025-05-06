FROM python:3.12-slim

# Cài Java (cho VnCoreNLP), và wget để tải model
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app và set làm thư mục làm việc
WORKDIR /app

# Copy toàn bộ source vào trong image
COPY . .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Tải model (sẽ không bị lặp nếu model đã tồn tại từ lần trước)
RUN python main.py download_only

# Mở cổng mặc định FastAPI
EXPOSE 8000

# Lệnh chạy app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
