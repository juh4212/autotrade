# main.py

import threading
from discord_bot import run_discord_bot, notify_discord
from scheduler import scheduler_job
import time
import logging
import os

# 로그 디렉토리 생성 (이미 scheduler.py에서 생성했지만, 안전을 위해 여기에도 추가)
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# 로깅 설정 (중복 설정 방지)
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
