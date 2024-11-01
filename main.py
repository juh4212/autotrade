# main.py

import threading
from discord_bot import run_discord_bot, notify_discord
from scheduler import scheduler_job
import time
import logging
import os

# 로그 디렉토리 생성 및 권한 설정
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
os.chmod(LOG_DIR, 0o777)  # 개발 단계에서는 777, 운영 환경에서는 최소 권한 부여

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'trading_bot.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def main():
    # Discord 봇 실행
    discord_thread = threading.Thread(target=run_discord_bot, daemon=True)
    discord_thread.start()
    logging.info("Discord 봇 스레드 시작")

    # 스케줄러 실행
    scheduler_thread = threading.Thread(target=scheduler_job, daemon=True)
    scheduler_thread.start()
    logging.info("스케줄러 스레드 시작")

    # 메인 스레드는 무한 루프로 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("프로그램 종료 요청")

if __name__ == "__main__":
    main()
