# main.py

import logging
import asyncio
from discord_bot import run_discord_bot, run_discord_bot_thread
from scheduler import start_scheduler

async def main():
    # 스케줄러 시작
    start_scheduler()

    # Discord 봇 시작
    await run_discord_bot()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # 필요 시 DEBUG로 변경
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("프로그램이 종료되었습니다.")
    except Exception as e:
        logging.error(f"메인 실행 중 에러 발생: {e}")
