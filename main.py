# main.py

import logging
import threading
from discord_bot import run_discord_bot_thread
from scheduler import scheduler_job

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # 필요 시 DEBUG로 변경
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 스케줄러를 별도의 스레드에서 실행
    scheduler_thread = threading.Thread(target=scheduler_job, daemon=True)
    scheduler_thread.start()
    logging.info("스케줄러가 별도의 스레드에서 시작되었습니다.")
    
    # Discord 봇 실행
    run_discord_bot_thread()
