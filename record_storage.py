# data_storage.py

from pymongo import MongoClient, errors
import os
import logging
import time

# 환경 변수에서 MongoDB URI 가져오기
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://mongo:27017/autotrade_db')

# MongoDB 클라이언트 설정
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

try:
    # 서버 연결 확인
    client.admin.command('ping')
    logging.info('MongoDB에 성공적으로 연결되었습니다.')
except errors.ConnectionFailure:
    logging.error('MongoDB에 연결할 수 없습니다: disconnected')
except Exception as e:
    logging.error(f'MongoDB 연결 중 에러 발생: {e}')

db = client.autotrade_db

# 컬렉션 설정
trade_records = db.trade_records
investment_performance = db.investment_performance

def save_trade_record(record):
    try:
        trade_records.insert_one(record)
        logging.info("거래 기록이 MongoDB에 저장되었습니다.")
    except errors.PyMongoError as e:
        logging.error(f"MongoDB에 거래 기록 저장 중 에러 발생: {e}")
    except Exception as e:
        logging.error(f"예기치 않은 에러 발생: {e}")

def save_investment_performance_record(record):
    try:
        investment_performance.insert_one(record)
        logging.info("투자 성과 기록이 MongoDB에 저장되었습니다.")
    except errors.PyMongoError as e:
        logging.error(f"MongoDB에 투자 성과 기록 저장 중 에러 발생: {e}")
    except Exception as e:
        logging.error(f"예기치 않은 에러 발생: {e}")
