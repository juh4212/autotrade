# 베이스 이미지로 Python 3.9를 사용
FROM python:3.9

# 작업 디렉토리 생성
WORKDIR /app

# 필요 파일들을 컨테이너에 복사
COPY requirements.txt /app/requirements.txt

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . /app

# 애플리케이션 시작 명령어 설정 (예: main.py가 실행 파일인 경우)
CMD ["python", "main.py"]
