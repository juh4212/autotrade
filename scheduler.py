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

        # 레버리지 사용 여부 결정 (Perpetuals 거래에서는 보통 레버리지 사용)
        is_leverage = True
        leverage = 5  # 원하는 레버리지로 설정

        # 포지션 크기 계산 (레버리지 포함 여부에 따라)
        order_quantity = calculate_position_size(equity, trade_percentage, leverage=leverage, is_leverage=is_leverage)

        # 'market_unit'은 Perpetuals 거래에서 필요 없으므로 제거

        # 'qty' 값을 정수로 설정 (계약 수량)
        qty = int(round(order_quantity))

        logging.info(f"시장 주문을 위한 계약 수량: {qty} (Perpetuals)")

        # AI 판단 로그
        logging.info(f"AI 판단: {side} 포지션, 퍼센티지: {trade_percentage}%, 레버리지: {leverage}x")

        # 실제로 주문할 수 있는지 확인
        # 계약 수량이 출금 가능 금액과 직접적으로 연관되지 않으므로, 필요에 따라 추가 로직 구현 가능
        # 여기서는 간단히 주문 가능 여부만 확인
        can_order = True  # 필요에 따라 실제 조건으로 변경

        if can_order:
            # 거래할 심볼과 방향 설정 (예: BTCUSDT, Buy/Sell)
            symbol = "BTCUSDT"  # 원하는 심볼로 변경

            # 주문 실행
            response = await place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market",
                category="linear",
                leverage=leverage
            )

            if response and response.get("retCode") == 0:
                logging.info(f"주문 성공: {response}")
            else:
                logging.error("주문 실패")
        else:
            logging.error("주문 조건을 만족하지 못했습니다.")
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
