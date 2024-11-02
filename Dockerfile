# Dockerfile

FROM python:3.9-slim

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# logs 디렉토리 생성 및 권한 설정
RUN mkdir -p logs && chmod -R 755 logs

# 비-루트 사용자 생성 및 소유권 설정
RUN adduser --disabled-password myuser
RUN chown -R myuser:myuser logs

# 비-루트 사용자로 전환
USER myuser

# 애플리케이션 실행
CMD ["python", "main.py"]  # main.py 실행
