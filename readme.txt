Xây dựng Docker Image
docker build -t fastapi-vncorenlp .

Chạy Container
docker run -d -p 8000:8000 --name fastapi-vncorenlp-container fastapi-vncorenlp

Truy cập API
http://localhost:8000/docs
