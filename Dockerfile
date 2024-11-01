# Dockerfile

# 베이스 이미지
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# logs 디렉토리 생성 및 권한 설정
RUN mkdir -p logs && chmod -R 777 logs

# 애플리케이션 실행
CMD ["python", "main.py"]
