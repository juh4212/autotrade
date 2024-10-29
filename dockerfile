# trading_bot/Dockerfile

# 베이스 이미지로 Python 3.11 슬림 버전 사용
FROM python:3.11-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 최신 버전으로 업그레이드
RUN pip install --upgrade pip

# 의존성 파일 복사 및 패키지 설치
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 애플리케이션 코드 복사
COPY trading_bot.py .

# 트레이딩 봇 실행
CMD ["python", "trading_bot.py"]
