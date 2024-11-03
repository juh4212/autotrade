# trade_execution.py

import logging
import os
from pybit.unified_trading import HTTP
import asyncio
import random

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

def determine_trade_percentage():
    """
    10~30% 사이의 정수 퍼센티지를 결정합니다.
    """
    percentage = random.randint(10, 30)
    logging.info(f"AI에 의해 결정된 거래 퍼센티지: {percentage}%")
    return percentage

def decide_long_or_short():
    """
    롱(Long) 또는 숏(Short)을 무작위로 결정합니다.
    """
    side = random.choice(["Buy", "Sell"])
    logging.info(f"거래 방향 결정: {side}")
    return side

def calculate_position_size(equity, percentage, leverage=5):
    """
    포지션 크기를 계산합니다.
    
    Parameters:
        equity (float): 총 자본 (equity)
        percentage (int): 진입 퍼센티지 (10~30%)
        leverage (int): 레버리지 (기본값: 5)
    
    Returns:
        float: 주문할 USDT 금액 (레버리지 포함)
    """
    position_usdt = (equity * percentage) / 100
    order_quantity = position_usdt * leverage
    logging.info(f"계산된 포지션 크기: {order_quantity} USDT (퍼센티지: {percentage}%, 레버리지: {leverage}x)")
    return order_quantity

async def place_order(symbol, side, qty, order_type="Market"):
    """
    Bybit에서 주문을 실행합니다.
    
    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        side (str): 주문 방향 ("Buy" 또는 "Sell")
        qty (float): 주문할 수량 (USDT 기준, 레버리지 포함)
        order_type (str): 주문 유형 (기본값: "Market")
    
    Returns:
        dict: 주문 응답
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        # 주문 실행 (레버리지는 이미 포지션 크기에 포함됨)
        response = await asyncio.to_thread(
            bybit_client.order.create,  # 수정된 메서드 이름
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
