# trade_execution.py

import logging
import os
from pybit.unified_trading import HTTP  # pybit v5에서 HTTP 클래스 임포트

# 환경 변수에서 Bybit API 키 및 시크릿 가져오기
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Bybit 클라이언트 초기화
if BYBIT_API_KEY and BYBIT_API_SECRET:
    bybit_client = HTTP(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET
    )
    logging.info("Bybit 클라이언트가 초기화되었습니다.")
else:
    bybit_client = None
    logging.error("Bybit API 키 또는 시크릿이 설정되지 않았습니다.")

def place_order(symbol, side, qty, order_type="Market"):
    """
    Bybit에서 주문을 실행합니다.
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.place_active_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            time_in_force="GoodTillCancel"
        )
        logging.info(f"{side} 주문이 실행되었습니다: {response}")
        return response
    except Exception as e:
        logging.error(f"주문 실행 중 에러 발생: {e}")
        return None
