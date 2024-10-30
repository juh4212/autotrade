import os
import logging
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from datetime import datetime
from pybit.unified_trading import HTTP  # Bybit v5 API를 사용 중임을 가정
import threading  # 스레드 안전성을 위한 라이브러리 추가

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
        bybit = HTTP(api_key=api_key, api_secret=api_secret)
        logger.info("Bybit API 연결 완료!")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

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
    # 추가적인 검증 로직을 여기에 추가할 수 있습니다.
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
        order_book = get_order_book(bybit)
        if not order_book:
            logger.error("오더북 데이터를 가져오지 못했습니다.")
            return

        # 3. 일별 OHLCV 데이터 조회 및 기술 지표 추가
        daily_ohlcv = get_daily_ohlcv(bybit)
        if not daily_ohlcv:
            logger.error("일별 OHLCV 데이터를 가져오지 못했습니다.")
            return

        # 4. 시간별 OHLCV 데이터 조회 및 기술 지표 추가
        hourly_ohlcv = get_hourly_ohlcv(bybit)
        if not hourly_ohlcv:
            logger.error("시간별 OHLCV 데이터를 가져오지 못했습니다.")
            return

        # 5. 공포 탐욕 지수 조회
        fear_greed_index = get_fear_greed_index()
        if fear_greed_index is None:
            logger.error("공포 탐욕 지수를 가져오지 못했습니다.")
            return

        # 6. 최신 비트코인 뉴스 헤드라인 가져오기
        news_headlines = get_latest_news_headlines()
        if not news_headlines:
            logger.error("뉴스 헤드라인을 가져오지 못했습니다.")
            return

        # 7. AI를 사용하여 트레이딩 결정 요청
        trading_decision = request_ai_trading_decision(
            balance_data,
            order_book,
            daily_ohlcv,
            hourly_ohlcv,
            fear_greed_index,
            news_headlines
        )
        if not trading_decision:
            logger.error("AI로부터 트레이딩 결정을 받지 못했습니다.")
            return

        # 8. AI 응답 파싱 및 결정 실행
        execute_trading_decision(bybit, collection, trading_decision, balance_data)

    except Exception as e:
        handle_error(e, "ai_trading 함수")

# 임시 도우미 함수 구현 (추후 단계에서 실제 구현 필요)
def get_account_balance(bybit):
    # 임시 구현
    return {"equity": 1000.0, "available_to_withdraw": 800.0}

def get_order_book(bybit):
    # 임시 구현
    return {"bids": [], "asks": []}

def get_daily_ohlcv(bybit):
    # 임시 구현
    return []

def get_hourly_ohlcv(bybit):
    # 임시 구현
    return []

def get_fear_greed_index():
    # 임시 구현
    return 50

def get_latest_news_headlines():
    # 임시 구현
    return ["Bitcoin hits new all-time high!"]

def request_ai_trading_decision(balance_data, order_book, daily_ohlcv, hourly_ohlcv, fear_greed_index, news_headlines):
    # 임시 구현
    return {"action": "hold"}

def execute_trading_decision(bybit, collection, trading_decision, balance_data):
    # 임시 구현
    logger.info(f"Executing trading decision: {trading_decision}")

if __name__ == "__main__":
    # MongoDB와 Bybit 연결 설정
    trades_collection = setup_mongodb()
    bybit = setup_bybit()

    # 트레이딩 작업을 테스트하기 위해 job 함수 호출
    job(bybit, trades_collection)
