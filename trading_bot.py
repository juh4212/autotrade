import os
import logging
from pymongo import MongoClient
from pybit.unified_trading import HTTP  # Bybit v5에 맞는 통합 API
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
print("환경 변수 로드 완료")  # 로드 여부 확인

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# MongoDB 설정 및 연결
def setup_mongodb():
    print("MongoDB 설정 시작")  # 설정 시작 여부 확인
    try:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            logger.critical("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
            raise ValueError("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        logger.info("MongoDB 연결 완료!")
        return trades_collection
    except Exception as e:
        logger.critical(f"MongoDB 연결 오류: {e}")
        raise

# Bybit API 설정
def setup_bybit():
    print("Bybit API 설정 시작")  # 설정 시작 여부 확인
    try:
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.critical("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
            raise ValueError("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
        bybit = HTTP(api_key=api_key, api_secret=api_secret)
        logger.info("Bybit API 연결 완료!")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

if __name__ == "__main__":
    # MongoDB와 Bybit 연결 테스트
    setup_mongodb()
    setup_bybit()
