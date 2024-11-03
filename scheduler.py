# scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
from data_collection import get_wallet_balance
from trade_execution import determine_trade_percentage, calculate_position_size, place_order
from datetime import datetime  # 추가

scheduler = AsyncIOScheduler()

async def execute_trade():
    """
    거래를 실행하는 스케줄된 작업.
    """
    logging.info("거래 작업 시작")

    # 잔고 정보 가져오기
    balance_info = await asyncio.to_thread(get_wallet_balance)

    if balance_info:
        equity = balance_info.get("equity", 0)
        available_to_withdraw = balance_info.get("available_to_withdraw", 0)

        logging.info(f"Total Equity: {equity} USDT")
        logging.info(f"Available to Withdraw: {available_to_withdraw} USDT")

        # AI를 사용하여 퍼센티지 결정
        trade_percentage = determine_trade_percentage()

        # 포지션 크기 계산
        position_size = calculate_position_size(equity, trade_percentage, leverage=5)

        # 실제로 주문할 수 있는지 확인
        if position_size <= available_to_withdraw:
            # 거래할 심볼과 방향 설정 (예: BTCUSDT, Buy/Sell)
            symbol = "BTCUSDT"  # 원하는 심볼로 변경
            side = "Buy"  # 또는 "Sell"

            # 주문 실행
            response = await place_order(symbol, side, position_size, leverage=5, order_type="Market")

            if response and response.get("retCode") == 0:
                logging.info(f"주문 성공: {response}")
            else:
                logging.error("주문 실패")
        else:
            logging.error("포지션 크기가 출금 가능 금액보다 큽니다.")
    else:
        logging.error("잔고 정보를 가져오지 못했습니다.")

def start_scheduler():
    """
    스케줄러를 설정하고 시작합니다.
    """
    # execute_trade를 즉시 실행
    scheduler.add_job(execute_trade, 'date', run_date=datetime.now())

    # execute_trade를 10분마다 반복 실행
    scheduler.add_job(execute_trade, 'interval', minutes=10)

    scheduler.start()
    logging.info("스케줄러가 시작되었습니다.")
