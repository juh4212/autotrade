# scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
from data_collection import get_wallet_balance
from trade_execution import determine_trade_percentage, decide_long_or_short, calculate_position_size, place_order
from datetime import datetime
import asyncio

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

        # 퍼센티지 결정 (10~30% 정수)
        trade_percentage = determine_trade_percentage()

        # 롱 또는 숏 결정
        side = decide_long_or_short()

        # 레버리지 사용 여부 결정 (예: 무조건 사용하지 않거나, 조건에 따라 결정)
        is_leverage = False  # 현재는 Spot 트레이딩만 사용하므로 False

        # 포지션 크기 계산 (레버리지 포함 여부에 따라)
        order_quantity = calculate_position_size(equity, trade_percentage, leverage=5, is_leverage=is_leverage)

        # AI 판단 로그
        logging.info(f"AI 판단: {side} 포지션, 퍼센티지: {trade_percentage}%, 레버리지: {'5x' if is_leverage else '1x'}")

        # 실제로 주문할 수 있는지 확인
        # 'market_unit'이 'value'인 경우, 'order_quantity'은 USDT 금액
        # 따라서, 소수점 2자리로 제한
        order_quantity = round(order_quantity, 2)

        if order_quantity <= available_to_withdraw * 5:  # 레버리지 x5 고려 (만약 레버리지 사용 시)
            # 거래할 심볼과 방향 설정 (예: BTCUSDT, Buy/Sell)
            symbol = "BTCUSDT"  # 원하는 심볼로 변경

            # Spot 시장 주문 시 market_unit 설정
            market_unit = "value" if side == "Buy" else "qty"

            # 주문 실행
            response = await place_order(
                symbol=symbol,
                side=side,
                qty=order_quantity,
                order_type="Market",
                category="spot",
                market_unit=market_unit
            )

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
