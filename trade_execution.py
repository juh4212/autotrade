# trade_execution.py

import logging
import os
from pybit.unified_trading import HTTP
import asyncio
import random
from ai_judgment import get_ai_decision  # ai_judgment.py의 함수 임포트
from data_collection import get_recent_trades, get_current_market_data  # 데이터 수집 함수 임포트
from discord_bot import post_reflection  # Discord에 반성 내용 게시 함수 임포트

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 필요 시 DEBUG로 변경
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

def calculate_position_size(equity, percentage, leverage=5, is_leverage=True):
    """
    포지션 크기를 계산합니다.
    
    Parameters:
        equity (float): 총 자본 (equity)
        percentage (int): 진입 퍼센티지 (10~30%)
        leverage (int): 레버리지 (기본값: 5)
        is_leverage (bool): 레버리지 사용 여부
    
    Returns:
        float: 주문할 수량 (레버리지 포함)
    """
    position_usdt = (equity * percentage) / 100
    if is_leverage:
        order_quantity = position_usdt * leverage
        logging.info(f"레버리지를 사용하여 계산된 포지션 크기: {order_quantity} USDT (퍼센티지: {percentage}%, 레버리지: {leverage}x)")
    else:
        order_quantity = position_usdt
        logging.info(f"레버리지를 사용하지 않고 계산된 포지션 크기: {order_quantity} USDT (퍼센티지: {percentage}%)")
    
    return order_quantity

async def place_order(symbol, side, qty, order_type="Market", category="linear", leverage=5):
    """
    Bybit에서 주문을 실행합니다.
    
    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        side (str): 주문 방향 ("Buy" 또는 "Sell")
        qty (int): 주문할 계약 수량 (정수)
        order_type (str): 주문 유형 (기본값: "Market")
        category (str): 제품 유형 (기본값: "linear" for USDT Perpetuals)
        leverage (int): 레버리지 설정 (기본값: 5)
    
    Returns:
        dict: 주문 응답
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        # 레버리지 설정
        if category == "linear":
            leverage_params = {}
            if side == "Buy":
                leverage_params["buyLeverage"] = leverage
            elif side == "Sell":
                leverage_params["sellLeverage"] = leverage

            if leverage_params:
                response_leverage = await asyncio.to_thread(
                    bybit_client.set_leverage,
                    symbol=symbol,
                    **leverage_params
                )
                logging.info(f"레버리지 설정 응답: {response_leverage}")

        # 주문 파라미터 설정
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(int(qty)),               # 계약 수량은 정수로 설정
            "price": "0",                       # Market 주문이므로 price는 0으로 설정
            "timeInForce": "GoodTillCancel",
            "orderLinkId": f"auto-trade-{int(random.random() * 100000)}",
            "isLeverage": 1                     # 레버리지 사용
        }

        if order_type == "Limit":
            params["price"] = str(int(params["price"]))  # 가격을 정수로 설정

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

async def execute_trade():
    """
    AI의 판단을 받아 매매를 실행하는 함수
    """
    # 데이터 수집
    trades_df = await asyncio.to_thread(get_recent_trades)  # 최근 거래 내역 가져오기
    current_market_data = await asyncio.to_thread(get_current_market_data)  # 현재 시장 데이터 가져오기

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

    if decision_type.lower() == "buy":
        # 예시: USDT 잔고 조회 및 매수 실행
        usdt_balance = await asyncio.to_thread(get_usdt_balance)  # USDT 잔고 조회 함수 필요
        if usdt_balance is None:
            logging.error("USDT 잔고 조회 실패.")
            return
        buy_amount = usdt_balance * (percentage / 100) * 0.9995  # 수수료 고려
        if buy_amount > 5000:
            logging.info(f"Buy Order Executed: {percentage}% of available USDT")
            response = await place_order(
                symbol="BTCUSDT",
                side="Buy",
                qty=calculate_position_size(usdt_balance, percentage, leverage=5, is_leverage=True)
            )
            if response and response.get("retCode") == 0:
                logging.info(f"Buy order executed successfully: {response}")
            else:
                logging.error("Buy order failed.")
        else:
            logging.warning("Buy Order Failed: Insufficient USDT (less than 5000 USDT)")

    elif decision_type.lower() == "sell":
        # 예시: 보유 중인 BTC 조회 및 매도 실행
        btc_balance = await asyncio.to_thread(get_btc_balance)  # BTC 잔고 조회 함수 필요
        if btc_balance is None:
            logging.error("BTC 잔고 조회 실패.")
            return
        sell_amount = btc_balance * (percentage / 100)
        current_price = await asyncio.to_thread(get_current_price, "BTCUSDT")  # 현재 가격 조회 함수 필요
        if sell_amount * current_price > 5000:
            logging.info(f"Sell Order Executed: {percentage}% of held BTC")
            response = await place_order(
                symbol="BTCUSDT",
                side="Sell",
                qty=calculate_position_size(current_price, percentage, leverage=5, is_leverage=True)
            )
            if response and response.get("retCode") == 0:
                logging.info(f"Sell order executed successfully: {response}")
            else:
                logging.error("Sell order failed.")
        else:
            logging.warning("Sell Order Failed: Insufficient BTC (less than 5000 USDT worth)")

    elif decision_type.lower() == "hold":
        logging.info("Decision is to hold. No action taken.")
    else:
        logging.error("Invalid decision received from AI.")
        return

    # AI의 반성 내용 Discord에 게시
    reflection = f"Decision: {decision_type.upper()}, Percentage: {percentage}%, Reason: {reason}"
    await post_reflection(reflection)

    # 추가 로직: 주문 후 기록 저장 등
