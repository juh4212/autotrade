import os
import logging
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from datetime import datetime
from pybit.unified_trading import HTTP  # Bybit v5 API 사용
import threading
import pandas as pd
import ta  # 기술 지표 계산을 위한 라이브러리
import openai  # OpenAI API 사용
import re  # 정규 표현식 사용
import json  # JSON 파싱을 위한 라이브러리

# 환경 변수 로드
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB 설정 및 연결
def setup_mongodb():
    mongo_uri = os.getenv("MONGODB_URI")
    print("MongoDB URI:", mongo_uri)  # URI 확인을 위해 출력
    try:
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        # 필요한 인덱스 생성 (예: timestamp 인덱스)
        trades_collection.create_index([('timestamp', ASCENDING)])
        logger.info("MongoDB 연결 및 초기화 완료!")
        return trades_collection
    except Exception as e:
        logger.critical(f"MongoDB 연결 오류: {e}")
        raise

# Bybit API 설정
def setup_bybit():
    try:
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.critical("Bybit API 키가 설정되지 않았습니다.")
            raise ValueError("Bybit API 키가 누락되었습니다.")
        bybit = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            endpoint="https://api.bybit.com"
        )
        logger.info("Bybit API 연결 완료!")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

# OpenAI API 설정
def setup_openai():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.critical("OpenAI API 키가 설정되지 않았습니다.")
        raise ValueError("OpenAI API 키가 누락되었습니다.")
    openai.api_key = openai_api_key
    logger.info("OpenAI API 설정 완료!")

# 도우미 함수 정의

def get_current_timestamp():
    """
    현재 UTC 시간을 ISO 형식으로 반환합니다.
    """
    return datetime.utcnow().isoformat()

def validate_balance_data(balance_data):
    """
    잔고 데이터의 유효성을 검증합니다.
    """
    if not balance_data:
        logger.error("잔고 데이터가 비어 있습니다.")
        return False
    required_keys = ["equity", "available_to_withdraw"]
    for key in required_keys:
        if key not in balance_data:
            logger.error(f"잔고 데이터에 '{key}' 키가 없습니다.")
            return False
    return True

def handle_error(e, context=""):
    """
    공통 에러 핸들링 함수.
    """
    if context:
        logger.error(f"{context} 오류: {e}")
    else:
        logger.error(f"오류 발생: {e}")

def log_event(message, level="info"):
    """
    특정 이벤트를 로그로 기록하는 함수.
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.debug(message)

# MongoDB에 잔고 기록
def log_balance_to_mongodb(collection, balance_data):
    if not validate_balance_data(balance_data):
        return
    balance_record = {
        "timestamp": get_current_timestamp(),
        "balance_data": balance_data
    }
    try:
        collection.insert_one(balance_record)
        logger.info("계좌 잔고가 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        handle_error(e, "MongoDB에 계좌 잔고 저장")

# 현재 포지션 조회 함수 추가
def get_current_position(bybit, symbol="BTCUSDT"):
    """
    현재 포지션을 조회하는 함수.

    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")

    Returns:
        str: 현재 포지션 ('long', 'short', 'none') 또는 None
    """
    try:
        response = bybit.get_positions(
            category="linear",
            symbol=symbol
        )
        logger.debug(f"get_current_position API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"포지션 조회 실패: {response['retMsg']}")
            return None

        positions = response.get('result', {}).get('list', [])
        if not positions:
            logger.info("현재 포지션이 없습니다.")
            return "none"

        position = positions[0]
        side = position.get('side')
        if side == "Buy":
            return "long"
        elif side == "Sell":
            return "short"
        else:
            return "none"
    except Exception as e:
        handle_error(e, "get_current_position 함수")
        return None

# 글로벌 변수 및 락 설정
trading_in_progress = False
trading_lock = threading.Lock()

def job(bybit, collection):
    """
    스케줄링된 트레이딩 작업을 수행하는 함수.
    """
    global trading_in_progress
    with trading_lock:
        if trading_in_progress:
            logger.warning("이미 트레이딩 작업이 진행 중입니다. 현재 작업을 건너뜁니다.")
            return
        trading_in_progress = True

    try:
        logger.info("트레이딩 작업 시작...")
        ai_trading(bybit, collection)
        logger.info("트레이딩 작업 완료.")
    except Exception as e:
        handle_error(e, "job 함수")
    finally:
        with trading_lock:
            trading_in_progress = False

# ai_trading 함수 정의
def ai_trading(bybit, collection):
    """
    AI 기반 트레이딩 로직을 수행하는 함수.
    """
    try:
        # 1. 현재 잔고 조회
        balance_data = get_account_balance(bybit)
        if not balance_data:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return

        # 2. 오더북 데이터 조회
        order_book = get_order_book(bybit, symbol="BTCUSDT", category="spot", limit=200)
        if not order_book:
            logger.error("오더북 데이터를 가져오지 못했습니다.")
            return

        # 3. 일별 OHLCV 데이터 조회 및 기술 지표 추가
        daily_ohlcv = get_daily_ohlcv(bybit, symbol="BTCUSDT", interval="D", limit=100)
        if daily_ohlcv is None:
            logger.error("일별 OHLCV 데이터를 가져오지 못했습니다.")
            return

        # 4. 시간별 OHLCV 데이터 조회 및 기술 지표 추가
        hourly_ohlcv = get_hourly_ohlcv(bybit, symbol="BTCUSDT", interval="60", limit=100)
        if hourly_ohlcv is None:
            logger.error("시간별 OHLCV 데이터를 가져오지 못했습니다.")
            return

        # 5. 현재 포지션 조회
        current_position = get_current_position(bybit, symbol="BTCUSDT")
        if current_position is None:
            logger.error("현재 포지션을 조회하지 못했습니다.")
            return

        # 6. AI를 사용하여 트레이딩 결정 요청
        trading_decision = request_ai_trading_decision(
            collection,
            balance_data,
            daily_ohlcv,
            hourly_ohlcv
        )
        if not trading_decision:
            logger.error("AI로부터 트레이딩 결정을 받지 못했습니다.")
            return

        # 7. AI 응답 파싱 및 결정 실행
        execute_trading_decision(bybit, collection, trading_decision, balance_data, current_position)

    except Exception as e:
        handle_error(e, "ai_trading 함수")

# Bybit 계좌 잔고 조회
def get_account_balance(bybit):
    try:
        wallet_balance = bybit.get_wallet_balance(accountType="CONTRACT")
        logger.info("Bybit API 응답 데이터: %s", wallet_balance)  # 전체 응답 데이터 출력
        if wallet_balance['retCode'] == 0 and 'result' in wallet_balance:
            account_list = wallet_balance['result'].get('list', [])
            if account_list:
                account_info = account_list[0]
                coin_balances = account_info.get('coin', [])
                usdt_balance = next((coin for coin in coin_balances if coin['coin'] == 'USDT'), None)
                if usdt_balance:
                    equity = float(usdt_balance.get('equity', 0))
                    available_to_withdraw = float(usdt_balance.get('availableToWithdraw', 0))
                    logger.info(f"USDT 전체 자산: {equity}, 사용 가능한 자산: {available_to_withdraw}")
                    return {
                        "equity": equity,
                        "available_to_withdraw": available_to_withdraw
                    }
                else:
                    logger.error("USDT 잔고 데이터를 찾을 수 없습니다.")
                    return None
            else:
                logger.error("계정 리스트가 비어 있습니다.")
                return None
        else:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return None
    except Exception as e:
        logger.error(f"Bybit 잔고 조회 오류: {e}")
        return None

def get_order_book(bybit, symbol="BTCUSDT", category="spot", limit=200):
    """
    Bybit API를 사용하여 오더북 데이터를 가져오는 함수.

    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")
        category (str): 제품 유형 (예: "spot", "linear", "inverse", "option")
        limit (int): 각 bid와 ask의 제한 크기

    Returns:
        dict: 오더북 데이터 또는 None
    """
    try:
        response = bybit.get_orderbook(
            category=category,
            symbol=symbol,
            limit=limit
        )
        logger.debug(f"get_order_book API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"오더북 데이터 조회 실패: {response['retMsg']}")
            return None

        order_book = response.get('result', {})
        if not order_book:
            logger.error("오더북 데이터가 비어 있습니다.")
            return None

        return order_book
    except AttributeError as ae:
        logger.exception(f"get_order_book 함수에서 AttributeError 발생: {ae}")
        return None
    except Exception as e:
        logger.exception(f"get_order_book 함수에서 예외 발생: {e}")
        return None

def get_daily_ohlcv(bybit, symbol="BTCUSDT", interval="D", limit=100):
    """
    Bybit API를 사용하여 일별 OHLCV 데이터를 가져오는 함수.

    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름
        interval (str): 시간 간격 ("1" "3" "5" "15" "30" "60" "120" "240" "360" "720" "D" "W" "M")
        limit (int): 가져올 데이터의 개수

    Returns:
        pandas.DataFrame: OHLCV 데이터프레임 또는 None
    """
    try:
        response = bybit.get_kline(
            category="spot",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        logger.debug(f"get_daily_ohlcv API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"OHLCV 데이터 조회 실패: {response['retMsg']}")
            return None

        ohlcv_data = response.get('result', {}).get('list', [])
        if not ohlcv_data:
            logger.error("OHLCV 데이터가 비어 있습니다.")
            return None

        # 데이터 컬럼 지정
        df = pd.DataFrame(ohlcv_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['start'], unit='s')
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 기술 지표 추가
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()

        logger.info("일별 OHLCV 데이터 조회 및 처리 완료.")
        return df
    except Exception as e:
        handle_error(e, "get_daily_ohlcv 함수")
        return None

def get_hourly_ohlcv(bybit, symbol="BTCUSDT", interval="60", limit=100):
    """
    Bybit API를 사용하여 시간별 OHLCV 데이터를 가져오는 함수.

    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름
        interval (str): 시간 간격
        limit (int): 가져올 데이터의 개수

    Returns:
        pandas.DataFrame: OHLCV 데이터프레임 또는 None
    """
    try:
        response = bybit.get_kline(
            category="spot",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        logger.debug(f"get_hourly_ohlcv API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"OHLCV 데이터 조회 실패: {response['retMsg']}")
            return None

        ohlcv_data = response.get('result', {}).get('list', [])
        if not ohlcv_data:
            logger.error("OHLCV 데이터가 비어 있습니다.")
            return None

        # 데이터 컬럼 지정
        df = pd.DataFrame(ohlcv_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['start'], unit='s')
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 기술 지표 추가
        df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        logger.info("시간별 OHLCV 데이터 조회 및 처리 완료.")
        return df
    except Exception as e:
        handle_error(e, "get_hourly_ohlcv 함수")
        return None

# 이하 필요한 함수들을 추가로 작성해주세요 (예: request_ai_trading_decision, execute_trading_decision 등)

if __name__ == "__main__":
    # MongoDB, Bybit 및 OpenAI 연결 설정
    trades_collection = setup_mongodb()
    bybit = setup_bybit()
    setup_openai()

    # 트레이딩 작업을 테스트하기 위해 job 함수 호출
    job(bybit, trades_collection)
