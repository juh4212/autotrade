# trade_execution.py

import logging
import os
import asyncio
import random
import pandas as pd
import json
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
from ai_judgment import get_ai_decision  # ai_judgment.py에서 AI 판단 함수 임포트

# 로깅 설정 (이미 설정되어 있다면 중복 설정을 피하세요)
logging.basicConfig(
    level=logging.DEBUG,  # 디버깅을 위해 DEBUG 레벨로 설정
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 환경 변수에서 Bybit API 키 및 시크릿 가져오기
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('USE_TESTNET', 'False').lower() in ['true', '1', 't']

# API 엔드포인트 설정
if USE_TESTNET:
    BASE_URL = 'https://api-testnet.bybit.com'
else:
    BASE_URL = 'https://api.bybit.com'

def generate_signature(api_secret, method, endpoint, params):
    """
    서명을 생성하는 함수.

    Parameters:
        api_secret (str): API 시크릿 키
        method (str): HTTP 메서드 ('GET', 'POST' 등)
        endpoint (str): API 요청 경로 (예: '/v5/position/list')
        params (dict): 요청에 사용될 파라미터

    Returns:
        str: 생성된 서명
    """
    # 타임스탬프 (밀리초)
    timestamp = int(time.time() * 1000)
    
    # 요청 파라미터에 타임스탬프 추가
    params['timestamp'] = timestamp

    # 쿼리 문자열 정렬
    sorted_params = urlencode(sorted(params.items()))
    
    # 서명할 문자열 생성
    pre_sign = method.upper() + endpoint + sorted_params
    
    # HMAC-SHA256 서명 생성
    signature = hmac.new(
        api_secret.encode('utf-8'),
        pre_sign.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature, timestamp

def get_position_list(category='linear', symbol='BTCUSDT'):
    """
    Bybit v5 API를 사용하여 포지션 정보를 가져옵니다.

    Parameters:
        category (str): 카테고리 ('linear' 또는 'inverse')
        symbol (str): 거래 심볼 (예: 'BTCUSDT')

    Returns:
        list: 열린 포지션 목록 또는 None
    """
    endpoint = '/v5/position/list'
    method = 'GET'
    url = BASE_URL + endpoint

    # 요청 파라미터
    params = {
        'category': category,
        'symbol': symbol
    }

    # 서명 생성
    signature, timestamp = generate_signature(API_SECRET, method, endpoint, params)

    # 헤더 설정
    headers = {
        'X-BAPI-API-KEY': API_KEY,
        'X-BAPI-TIMESTAMP': str(timestamp),
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        data = response.json()
        
        # 응답 데이터 로깅
        logger.debug("API 응답 데이터: %s", json.dumps(data, indent=4, ensure_ascii=False))
        
        if data.get('retCode') == 0:
            positions = data['result']['list']
            if positions:
                logger.info(f"{symbol.upper()}의 열린 포지션을 가져왔습니다.")
                return positions
            else:
                logger.info(f"{symbol.upper()}에 열린 포지션이 없습니다.")
                return []
        else:
            logger.error(f"API 오류: {data.get('retMsg')}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP 요청 오류: {e}")
        return None

def get_market_data(symbol):
    """
    Bybit에서 지정된 심볼의 시장 데이터를 가져옵니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")

    Returns:
        dict: 시장 데이터
    """
    # 예제: 주문 책(Order Book) 데이터 가져오기
    endpoint = '/v5/market/orderbook'
    method = 'GET'
    url = BASE_URL + endpoint

    params = {
        'category': 'linear',
        'symbol': symbol.upper()
    }

    signature, timestamp = generate_signature(API_SECRET, method, endpoint, params)

    headers = {
        'X-BAPI-API-KEY': API_KEY,
        'X-BAPI-TIMESTAMP': str(timestamp),
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug("get_market_data 응답 데이터: %s", json.dumps(data, indent=4, ensure_ascii=False))
        
        if data.get('retCode') == 0:
            orderbook = data['result']
            logger.info(f"{symbol.upper()}의 시장 데이터를 가져왔습니다.")
            return orderbook
        else:
            logger.error(f"시장 데이터를 가져오는 중 에러 발생: {data.get('retMsg')}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"시장 데이터를 가져오는 중 HTTP 오류: {e}")
        return None

def get_recent_trades(symbol, limit=50):
    """
    Bybit에서 지정된 심볼의 최근 거래 내역을 가져옵니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        limit (int): 가져올 거래 수 (기본값: 50)

    Returns:
        list: 최근 거래 내역
    """
    endpoint = '/v5/market/trading-records'
    method = 'GET'
    url = BASE_URL + endpoint

    params = {
        'category': 'linear',
        'symbol': symbol.upper(),
        'limit': limit
    }

    signature, timestamp = generate_signature(API_SECRET, method, endpoint, params)

    headers = {
        'X-BAPI-API-KEY': API_KEY,
        'X-BAPI-TIMESTAMP': str(timestamp),
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug("get_recent_trades 응답 데이터: %s", json.dumps(data, indent=4, ensure_ascii=False))
        
        if data.get('retCode') == 0:
            trades = data['result']['list']
            logger.info(f"{symbol.upper()}의 최근 거래 내역을 가져왔습니다.")
            return trades
        else:
            logger.error(f"최근 거래 내역을 가져오는 중 에러 발생: {data.get('retMsg')}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"최근 거래 내역을 가져오는 중 HTTP 오류: {e}")
        return None

def get_kline_data(symbol, interval='15', limit=200):
    """
    Bybit에서 지정된 심볼의 캔들 차트 데이터를 가져옵니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        interval (str): 캔들 차트 간격 (예: '1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
        limit (int): 가져올 데이터 수 (기본값: 200)

    Returns:
        list: 캔들 차트 데이터
    """
    endpoint = '/v5/market/kline'
    method = 'GET'
    url = BASE_URL + endpoint

    params = {
        'category': 'linear',
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }

    signature, timestamp = generate_signature(API_SECRET, method, endpoint, params)

    headers = {
        'X-BAPI-API-KEY': API_KEY,
        'X-BAPI-TIMESTAMP': str(timestamp),
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug("get_kline_data 응답 데이터: %s", json.dumps(data, indent=4, ensure_ascii=False))
        
        if data.get('retCode') == 0:
            kline_data = data['result']['list']
            logger.info(f"{symbol.upper()}의 캔들 차트 데이터를 가져왔습니다.")
            return kline_data
        else:
            logger.error(f"캔들 차트 데이터를 가져오는 중 에러 발생: {data.get('retMsg')}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"캔들 차트 데이터를 가져오는 중 HTTP 오류: {e}")
        return None

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
    logger.info(f"계산된 포지션 크기: {order_quantity} USDT (퍼센티지: {percentage}%, 레버리지: {leverage}x)")
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
    endpoint = '/v5/order/create'
    method = 'POST'
    url = BASE_URL + endpoint

    # 타임스탬프
    timestamp = int(time.time() * 1000)

    # 요청 파라미터
    params = {
        'category': category,
        'symbol': symbol.upper(),
        'side': side.capitalize(),  # 'Buy' 또는 'Sell'
        'orderType': order_type,
        'qty': str(qty),
        'timeInForce': 'GoodTillCancel',
        'orderLinkId': f"auto-trade-{int(random.random() * 100000)}",
        'isLeverage': 1
    }

    # 쿼리 문자열 정렬
    sorted_params = urlencode(sorted(params.items()))

    # 서명할 문자열 생성
    pre_sign = method.upper() + '/v5/order/create' + sorted_params

    # HMAC-SHA256 서명 생성
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        pre_sign.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # 헤더 설정
    headers = {
        'X-BAPI-API-KEY': API_KEY,
        'X-BAPI-TIMESTAMP': str(timestamp),
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, json=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug("place_order 응답 데이터: %s", json.dumps(data, indent=4, ensure_ascii=False))
        
        if data.get('retCode') == 0:
            logger.info(f"{side.capitalize()} 주문이 성공적으로 실행되었습니다: {data}")
        else:
            logger.error(f"{side.capitalize()} 주문 실행 실패: {data.get('retMsg')}")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"주문 실행 중 HTTP 오류: {e}")
        return None
    except Exception as e:
        logger.error(f"주문 실행 중 예외 발생: {e}")
        return None

async def close_all_positions(symbol):
    """
    모든 열린 포지션을 청산합니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")

    Returns:
        bool: 청산 성공 여부
    """
    try:
        positions = get_position_list(category='linear', symbol=symbol)
        if positions:
            for position in positions:
                side = "Sell" if position['side'].lower() == "buy" else "Buy"
                qty = float(position['size'])
                logger.info(f"포지션 청산을 시도합니다: {side} {qty} {symbol.upper()}")
                response = await place_order(symbol, side, qty, order_type="Market", category="linear")
                if response and response.get('retCode') == 0:
                    logger.info(f"포지션 청산 주문이 성공적으로 실행되었습니다: {response}")
                else:
                    logger.error(f"포지션 청산 주문 실행 실패: {response.get('retMsg') if response else 'No Response'}")
            return True
        else:
            logger.info("열린 포지션이 없습니다. 청산할 필요가 없습니다.")
            return True
    except Exception as e:
        logger.error(f"포지션 청산 중 에러 발생: {e}")
        return False

async def execute_trade():
    """
    AI의 판단을 받아 매매를 실행하는 함수
    """
    # CONTRACT 계정 (파생상품 계정)에서 USDT 잔고 조회
    balance_info = get_account_balance(category='linear', symbol='BTCUSDT')
    if not balance_info:
        logger.error("잔고 정보를 가져오지 못했습니다.")
        return

    equity = balance_info.get("equity", 0)
    available_balance = balance_info.get("available_balance", 0)

    logger.debug(f"잔고 정보: equity={equity}, available_balance={available_balance}")

    if equity <= 0:
        logger.error("유효한 잔고가 없습니다.")
        return

    # 거래할 심볼 설정
    symbol = "BTCUSDT"  # 필요에 따라 변경

    # 열린 포지션 조회
    open_positions = get_position_list(category='linear', symbol=symbol)
    if open_positions is None:
        logger.error("열린 포지션을 가져오지 못했습니다.")
        return
    elif len(open_positions) > 0:
        logger.info(f"현재 {symbol.upper()}에 열린 포지션이 있습니다: {open_positions}")
        # 포지션 청산 시도
        success = await close_all_positions(symbol)
        if not success:
            logger.error("포지션 청산에 실패했습니다.")
            return
        else:
            logger.info("포지션을 성공적으로 청산했습니다.")
        # 청산 후 잔고 재조회
        balance_info = get_account_balance(category='linear', symbol=symbol)
        if not balance_info:
            logger.error("청산 후 잔고 정보를 가져오지 못했습니다.")
            return
        equity = balance_info.get("equity", 0)
        available_balance = balance_info.get("available_balance", 0)
        logger.debug(f"청산 후 잔고 정보: equity={equity}, available_balance={available_balance}")

    # 현재 시장 데이터 수집
    current_market_data = get_market_data(symbol)
    if not current_market_data:
        logger.error("시장 데이터를 가져오지 못했습니다.")
        return

    # 최근 거래 내역 가져오기
    trades = get_recent_trades(symbol)
    if not trades:
        logger.error("최근 거래 내역을 가져오지 못했습니다.")
        return

    trades_df = pd.DataFrame(trades)  # 실제 최근 거래 내역을 가져오는 로직으로 대체 필요

    # AI 판단 받아오기
    decision = get_ai_decision(trades_df, current_market_data)
    if not decision:
        logger.error("AI의 판단을 받아오지 못했습니다.")
        return

    decision_type = decision.get('decision')
    percentage = decision.get('percentage')
    reason = decision.get('reason')

    logger.debug(f"AI Decision: {decision_type}, Percentage: {percentage}, Reason: {reason}")

    if not decision_type or percentage is None:
        logger.error("AI 판단이 불완전합니다.")
        return

    logger.info(f"AI Decision: {decision_type.upper()}")
    logger.info(f"Percentage: {percentage}")
    logger.info(f"Reason: {reason}")

    leverage = 5        # 레버리지 설정
    trade_mode = 1      # 0: Cross Margin, 1: Isolated Margin

    # 마진 모드 및 레버리지 설정
    # Bybit v5 API에서는 이미 레버리지를 설정했으므로 추가 설정 필요 없음

    # 가격 및 수량 소수점 자릿수 가져오기
    price_precision, qty_precision = get_precisions(symbol)
    if price_precision is None or qty_precision is None:
        logger.error("가격 및 수량 소수점 자릿수를 가져오지 못했습니다.")
        return

    # 포지션 크기 계산
    qty = calculate_position_size(equity, percentage, leverage=leverage)

    # 수량을 소수점 자릿수에 맞게 반올림
    qty = round(qty, qty_precision)

    logger.debug(f"반올림된 주문 수량: {qty}")

    if decision_type.lower() == "buy":
        logger.info(f"매수 주문을 실행합니다. 수량: {qty}")
        # 매수 주문 실행
        response = await place_order(symbol, "Buy", qty, order_type="Market", category="linear")
    elif decision_type.lower() == "sell":
        logger.info(f"매도 주문을 실행합니다. 수량: {qty}")
        # 매도 주문 실행
        response = await place_order(symbol, "Sell", qty, order_type="Market", category="linear")
    elif decision_type.lower() == "hold":
        logger.info("보유 결정입니다. 주문을 실행하지 않습니다.")
    else:
        logger.error("AI로부터 유효하지 않은 결정이 전달되었습니다.")

    # 주문 실행 후 추가 작업 (예: 데이터베이스에 기록 저장 등)
