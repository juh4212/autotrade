# scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
from trade_execution import execute_trade
from datetime import datetime
import asyncio

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 필요 시 DEBUG로 변경
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

scheduler = AsyncIOScheduler()

async def ai_trading():
    """
    AI 기반 트레이딩을 실행하는 스케줄된 작업
    """
    logging.info("AI 트레이딩 작업 시작")
    await execute_trade()

def start_scheduler():
    """
    스케줄러를 설정하고 시작합니다.
    """
    # ai_trading을 즉시 실행
    scheduler.add_job(ai_trading, 'date', run_date=datetime.now())

    # ai_trading을 10분마다 반복 실행
    scheduler.add_job(ai_trading, 'interval', minutes=10)

    scheduler.start()
    logging.info("스케줄러가 시작되었습니다.")
