Xây dựng Docker Image
docker build -t ner-fastapi .

Chạy Container
docker run -d -p 8000:8000 --name ner-fastapi-container ner-fastapi

Truy cập API
http://localhost:8000/docs
