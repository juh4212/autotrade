import os
import time
import logging
import requests
from pymongo import MongoClient
from pybit.bybit_api import USDTPerpetual  # USDT 선물 거래에 맞는 모듈
import openai
import schedule
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import json
import re

# 환경 변수 로드
load_dotenv()

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

# Bybit 계좌 잔고 조회 (전체 응답 출력)
def get_account_balance(bybit):
    try:
        # 선물 계좌 잔고 확인 (account_type 파라미터 제거)
        wallet_balance = bybit.get_wallet_balance(coin="USDT")
        print("Bybit API 응답 데이터:", wallet_balance)  # 전체 응답 데이터 출력
        
        if 'result' in wallet_balance and wallet_balance['result']:
            usdt_total = wallet_balance['result']['USDT'].get('equity', 0)  # 총 잔고
            usdt_available = wallet_balance['result']['USDT'].get('available_balance', 0)  # 사용 가능한 잔고
            logger.info(f"USDT 총 잔고: {usdt_total}, 사용 가능한 잔고: {usdt_available}")
            return {"USDT_total": usdt_total, "USDT_available": usdt_available}
        else:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return None
    except Exception as e:
        logger.error(f"Bybit 잔고 조회 오류: {e}")
        return None

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

    # Bybit API를 사용하여 계좌 잔고 가져오기
    balance_data = get_account_balance(bybit)
    if balance_data:
        # MongoDB에 잔고 기록
        log_balance_to_mongodb(trades_collection, balance_data)
