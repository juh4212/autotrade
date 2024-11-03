# trade_execution.py

import logging
import os
import asyncio
import random
from pybit.unified_trading import HTTP
from ai_judgment import get_ai_decision  # ai_judgment.py에서 AI 판단 함수 임포트
from data_collection import get_wallet_balance, get_market_data  # 데이터 수집 함수 임포트
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # 디버깅을 위해 DEBUG 레벨로 설정
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 환경 변수에서 Bybit API 키 및 시크릿 가져오기
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('USE_TESTNET', 'False').lower() in ['true', '1', 't']

# Bybit 클라이언트 초기화
if BYBIT_API_KEY and BYBIT_API_SECRET:
    try:
        bybit_client = HTTP(
            testnet=USE_TESTNET,
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

def set_mode(symbol, trade_mode, leverage):
    """
    마진 모드와 레버리지를 설정합니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        trade_mode (int): 마진 모드 (0: Cross Margin, 1: Isolated Margin)
        leverage (int): 레버리지 설정 값
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return

    try:
        resp = bybit_client.switch_margin_mode(
            category='linear',
            symbol=symbol,
            tradeMode=trade_mode,
            buyLeverage=leverage,
            sellLeverage=leverage
        )
        logging.info(f"마진 모드 및 레버리지 설정 응답: {resp}")
    except Exception as err:
        logging.error(f"마진 모드 및 레버리지 설정 중 에러 발생: {err}")

def get_precisions(symbol):
    """
    지정된 심볼에 대한 가격과 수량의 소수점 자릿수를 가져옵니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")

    Returns:
        tuple: (price_precision, qty_precision)
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None, None

    try:
        response = bybit_client.get_instruments_info(
            category='linear',
            symbol=symbol
        )
        logging.debug(f"get_precisions 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            resp = response['result']['list'][0]
            price_tick_size = resp['priceFilter']['tickSize']
            if '.' in price_tick_size:
                price_precision = len(price_tick_size.split('.')[1].rstrip('0'))
            else:
                price_precision = 0

            qty_step = resp['lotSizeFilter']['qtyStep']
            if '.' in qty_step:
                qty_precision = len(qty_step.split('.')[1].rstrip('0'))
            else:
                qty_precision = 0

            logging.info(f"{symbol}의 가격 소수점 자릿수: {price_precision}, 수량 소수점 자릿수: {qty_precision}")
            return price_precision, qty_precision
        else:
            logging.error(f"상품 정보를 가져오는 중 에러 발생: {response['retMsg']}")
            return None, None
    except Exception as err:
        logging.error(f"가격 및 수량 소수점 자릿수 가져오기 중 예외 발생: {err}")
        return None, None

def calculate_position_size(equity, percentage, leverage=5):
    """
    포지션 크기를 계산합니다.

    Parameters:
        equity (float): 총 자본 (equity)
        percentage (int): 진입 퍼센티지 (예: 10-30)
        leverage (int): 레버리지 (기본값: 5)

    Returns:
        float: 주문할 계약 수량
    """
    position_usdt = (equity * percentage) / 100
    order_quantity = position_usdt * leverage
    logging.info(f"계산된 포지션 크기: {order_quantity} USDT (퍼센티지: {percentage}%, 레버리지: {leverage}x)")
    return order_quantity

async def place_order(symbol, side, qty, order_type="Market", category="linear"):
    """
    Bybit에서 주문을 실행합니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        side (str): 주문 방향 ("Buy" 또는 "Sell")
        qty (float): 주문할 계약 수량
        order_type (str): 주문 유형 (기본값: "Market")
        category (str): 제품 유형 ("linear" 또는 "inverse")

    Returns:
        dict: 주문 응답
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GoodTillCancel",
            "orderLinkId": f"auto-trade-{int(random.random() * 100000)}",
            "isLeverage": 1
        }

        response = await asyncio.to_thread(
            bybit_client.place_order,
            **params
        )
        logging.info(f"{side} 주문이 실행되었습니다: {response}")
        return response
    except Exception as e:
        logging.error(f"주문 실행 중 에러 발생: {e}")
        return None

async def execute_trade():
    """
    AI의 판단을 받아 매매를 실행하는 함수
    """
    # 잔고 정보 가져오기
    balance_info = await asyncio.to_thread(get_wallet_balance)
    if not balance_info:
        logging.error("잔고 정보를 가져오지 못했습니다.")
        return

    equity = balance_info.get("equity", 0)
    available_balance = balance_info.get("available_balance", 0)

    if equity <= 0:
        logging.error("유효한 잔고가 없습니다.")
        return

    # 거래할 심볼 설정
    symbol = "BTCUSDT"  # 필요에 따라 변경

    # 현재 시장 데이터 수집
    current_market_data = await asyncio.to_thread(get_market_data, symbol)
    if not current_market_data:
        logging.error("시장 데이터를 가져오지 못했습니다.")
        return

    # 최근 거래 내역 가져오기
    trades_df = pd.DataFrame()  # 실제 최근 거래 내역을 가져오는 로직으로 대체 필요

    # AI 판단 받아오기
    decision = get_ai_decision(trades_df, current_market_data)
    if not decision:
        logging.error("AI의 판단을 받아오지 못했습니다.")
        return

    decision_type = decision.get('decision')
    percentage = decision.get('percentage')
    reason = decision.get('reason')

    if not decision_type or percentage is None:
        logging.error("AI 판단이 불완전합니다.")
        return

    logging.info(f"AI Decision: {decision_type.upper()}")
    logging.info(f"Percentage: {percentage}")
    logging.info(f"Reason: {reason}")

    leverage = 5        # 레버리지 설정
    trade_mode = 1      # 0: Cross Margin, 1: Isolated Margin

    # 마진 모드 및 레버리지 설정
    set_mode(symbol, trade_mode, leverage)

    # 가격 및 수량 소수점 자릿수 가져오기
    price_precision, qty_precision = get_precisions(symbol)
    if price_precision is None or qty_precision is None:
        logging.error("가격 및 수량 소수점 자릿수를 가져오지 못했습니다.")
        return

    # 포지션 크기 계산
    qty = calculate_position_size(equity, percentage, leverage=leverage)

    # 수량을 소수점 자릿수에 맞게 반올림
    qty = round(qty, qty_precision)

    if decision_type.lower() == "buy":
        logging.info(f"매수 주문을 실행합니다. 수량: {qty}")
        # 매수 주문 실행
        response = await place_order(symbol, "Buy", qty, order_type="Market", category="linear")
    elif decision_type.lower() == "sell":
        logging.info(f"매도 주문을 실행합니다. 수량: {qty}")
        # 매도 주문 실행
        response = await place_order(symbol, "Sell", qty, order_type="Market", category="linear")
    elif decision_type.lower() == "hold":
        logging.info("보유 결정입니다. 주문을 실행하지 않습니다.")
    else:
        logging.error("AI로부터 유효하지 않은 결정이 전달되었습니다.")

    # 주문 실행 후 추가 작업 (예: 데이터베이스에 기록 저장 등)
