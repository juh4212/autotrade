# trade_execution.py

import logging
import os
from pybit.unified_trading import HTTP
import asyncio
import random

# 환경 변수에서 Bybit API 키 및 시크릿 가져오기
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('USE_TESTNET', 'False').lower() in ['true', '1', 't']

# Bybit 클라이언트 초기화
if BYBIT_API_KEY and BYBIT_API_SECRET:
    try:
        bybit_client = HTTP(
            testnet=USE_TESTNET,        # 테스트넷 사용 여부
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        logging.info("Bybit 클라이언트가 초기화되었습니다.")
    except Exception as e:
        bybit_client = None
        logging.error(f"Bybit 클라이언트 초기화 실패: {e}")
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
        float: 주문할 수량 (레버리지 포함)
    """
    position_usdt = (equity * percentage) / 100
    order_quantity = position_usdt * leverage
    logging.info(f"계산된 포지션 크기: {order_quantity} USDT (퍼센티지: {percentage}%, 레버리지: {leverage}x)")
    return order_quantity

async def place_order(symbol, side, qty, order_type="Market", category="spot", market_unit="value"):
    """
    Bybit에서 주문을 실행합니다.
    
    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        side (str): 주문 방향 ("Buy" 또는 "Sell")
        qty (float): 주문할 수량
        order_type (str): 주문 유형 (기본값: "Market")
        category (str): 제품 유형 (기본값: "spot", "linear", "inverse", "option")
        market_unit (str): Spot 시장 주문 시 qty 단위 ("value" 또는 "qty")
    
    Returns:
        dict: 주문 응답
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        # Spot 시장 주문 시 marketUnit 설정
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),               # Bybit API는 문자열로 qty를 요구할 수 있음
            "price": "0",                   # Market 주문이므로 price는 0으로 설정
            "timeInForce": "GoodTillCancel",
            "orderLinkId": f"auto-trade-{int(random.random() * 100000)}",
            "isLeverage": 0                # Spot 트레이딩
        }

        if category == "spot":
            params["marketUnit"] = market_unit  # Spot 시장 주문 시 marketUnit 설정

        # 주문 실행
        response = await asyncio.to_thread(
            bybit_client.place_order,
            **params
        )
        logging.info(f"{side} 주문이 실행되었습니다: {response}")
        return response
    except AttributeError as ae:
        logging.error(f"Bybit 클라이언트에 'place_order' 메서드가 없습니다: {ae}")
        return None
    except Exception as e:
        logging.error(f"주문 실행 중 에러 발생: {e}")
        return None
