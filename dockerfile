# 베이스 이미지 선택
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 코드 파일 복사
COPY . /app

# 환경 변수 로드
ENV PYTHONUNBUFFERED=1

# 컨테이너가 실행될 때 기본으로 실행할 명령어
CMD ["python", "trading_bot.py"]
