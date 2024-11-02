# main.py

import logging
import threading
from discord_bot import run_discord_bot_thread  # 수정된 임포트
from scheduler import scheduler_job

def start_scheduler():
    scheduler_job()

def start_discord_bot():
    run_discord_bot_thread()  # 코루틴을 올바르게 실행

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 스케줄러를 별도의 스레드에서 실행
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    logging.info("스케줄러가 별도의 스레드에서 시작되었습니다.")
    
    # Discord 봇 실행
    start_discord_bot()
