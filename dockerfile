# 베이스 이미지 선택 (최신 Python 버전 사용)
FROM python:3.13-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 비밀번호 없는 사용자 생성
RUN adduser --disabled-password --gecos '' appuser

# /app 디렉토리 소유권 변경
RUN chown -R appuser:appuser /app

# 비밀번호 없는 사용자로 변경
USER appuser

# 의존성 파일 복사
COPY --chown=appuser:appuser requirements.txt .

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY --chown=appuser:appuser . .

# 로그 디렉토리 생성 및 권한 설정
RUN mkdir -p logs && chmod 755 logs

# 실행 명령 설정
CMD ["python", "trading_bot.py"]
