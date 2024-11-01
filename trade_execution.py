# trade_execution.py

from bybit import bybit
import os
from dotenv import load_dotenv
from urllib.parse import urlencode
import jwt
import uuid
import hashlib
import hmac
import time

# 환경 변수 로드
load_dotenv()

BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Bybit 클라이언트 설정
client = bybit(test=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

def place_order(symbol, side, qty=None, price=None, order_type="Market"):
    """
    Bybit API를 사용하여 주문을 생성합니다.
    - side: "Buy" 또는 "Sell"
    - order_type: "Market" 또는 "Limit"
    """
    try:
        if order_type.lower() == "market":
            if side.lower() == "buy":
                order = client.Order.Order_new(
                    symbol=symbol,
                    side="Buy",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel"
                ).result()
            elif side.lower() == "sell":
                order = client.Order.Order_new(
                    symbol=symbol,
                    side="Sell",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel"
                ).result()
        elif order_type.lower() == "limit":
            if side.lower() == "buy":
                order = client.Order.Order_new(
                    symbol=symbol,
                    side="Buy",
                    order_type="Limit",
                    qty=qty,
                    price=price,
                    time_in_force="GoodTillCancel"
                ).result()
            elif side.lower() == "sell":
                order = client.Order.Order_new(
                    symbol=symbol,
                    side="Sell",
                    order_type="Limit",
                    qty=qty,
                    price=price,
                    time_in_force="GoodTillCancel"
                ).result()
        else:
            return {"error": "지원하지 않는 주문 유형입니다."}
        
        return order
    except Exception as e:
        print(f"주문 실행 에러: {e}")
        return {"error": str(e)}

def get_position(symbol):
    """
    현재 포지션을 조회합니다.
    """
    try:
        position = client.Positions.Positions_myPosition(symbol=symbol).result()
        return position
    except Exception as e:
        print(f"포지션 조회 에러: {e}")
        return {"error": str(e)}

# 테스트용 호출
if __name__ == "__main__":
    symbol = "BTCUSD"
    decision = "buy"  # 'buy', 'sell', 'close', 'hold'

    if decision == "buy":
        qty = 1  # 구매할 수량 설정
        order = place_order(symbol, "Buy", qty=qty, order_type="Market")
    elif decision == "sell":
        qty = 1  # 판매할 수량 설정
        order = place_order(symbol, "Sell", qty=qty, order_type="Market")
    elif decision == "close":
        position = get_position(symbol)
        if position and not position.get("error"):
            qty = position[0]['size']  # 현재 포지션 수량
            order = place_order(symbol, "Sell" if position[0]['side'] == "Buy" else "Buy", qty=qty, order_type="Market")
        else:
            order = {"error": "포지션을 조회할 수 없습니다."}
    else:
        order = None

    print(order)

