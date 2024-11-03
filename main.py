# main.py

import logging
import asyncio
from discord_bot import run_discord_bot_task
from scheduler import start_scheduler

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 필요 시 DEBUG로 변경
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def main():
    # 스케줄러 시작
    start_scheduler()

    # Discord 봇 시작
    loop = asyncio.get_event_loop()
    run_discord_bot_task(loop)

    # 이벤트 루프 유지
    await asyncio.Event().wait()  # 무한 대기

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("프로그램이 종료되었습니다.")
    except Exception as e:
        logging.error(f"메인 실행 중 에러 발생: {e}")
