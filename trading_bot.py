import os
import logging
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from datetime import datetime
from pybit.unified_trading import HTTP  # Bybit v5 API를 사용 중임을 가정

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

# MongoDB에 잔고 기록
def log_balance_to_mongodb(collection, balance_data):
    balance_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "balance_data": balance_data
    }
    try:
        collection.insert_one(balance_record)
        logger.info("계좌 잔고가 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"MongoDB에 계좌 잔고 저장 오류: {e}")

if __name__ == "__main__":
    # MongoDB와 Bybit 연결 설정
    trades_collection = setup_mongodb()
    bybit = setup_bybit()

    # Bybit API를 사용하여 계좌 잔고 가져오기 (추후 구현 예정)
    # balance_data = get_account_balance(bybit)
    # if balance_data:
    #     # MongoDB에 잔고 기록
    #     log_balance_to_mongodb(trades_collection, balance_data)
