# main.py

import logging
import asyncio
from discord_bot import run_discord_bot  # 변경: run_discord_bot을 비동기로 유지
from scheduler import start_scheduler

async def main():
    # 스케줄러 시작
    start_scheduler()

    # Discord 봇 시작
    asyncio.create_task(run_discord_bot())

    # 이벤트 루프이므로, 계속 실행 상태를 유지
    await asyncio.Event().wait()  # 무한 대기

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
