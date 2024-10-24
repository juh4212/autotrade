# 베이스 이미지 선택 (최신 Python 버전 사용)
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    build-essential

# 의존성 파일 복사
COPY requirements.txt .

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정 (필요한 경우)
# ENV BYBIT_API_KEY=your_bybit_api_key
# ENV BYBIT_API_SECRET=your_bybit_api_secret
# ENV OPENAI_API_KEY=your_openai_api_key
# ENV SERPAPI_API_KEY=your_serpapi_api_key

# 실행 명령 설정
CMD ["python", "trading_bot.py"]
