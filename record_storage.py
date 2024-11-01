# record_storage.py

from pymongo import MongoClient
import os
import json
from dotenv import load_dotenv
from datetime import datetime

# 환경 변수 로드
load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')

# MongoDB 클라이언트 설정
client = MongoClient(MONGODB_URI)
db = client.trading_db  # 데이터베이스 이름
trade_records_col = db.trade_records
investment_performance_col = db.investment_performance

def save_trade_record(symbol, side, qty, price, order_type, status, response):
    """
    거래 내역을 MongoDB에 저장합니다.
    """
    try:
        record = {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "order_type": order_type,
            "status": status,
            "response": response
        }
        trade_records_col.insert_one(record)
    except Exception as e:
        print(f"거래 기록 저장 에러: {e}")

def save_investment_performance(total_trades, profitable_trades, win_rate):
    """
    투자 성과를 MongoDB에 저장합니다.
    """
    try:
        performance = {
            "timestamp": datetime.utcnow(),
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate
        }
        investment_performance_col.insert_one(performance)
    except Exception as e:
        print(f"투자 성과 저장 에러: {e}")

# 테스트용 호출
if __name__ == "__main__":
    # 거래 기록 저장 테스트
    test_order = {
        "order_id": "123456",
        "symbol": "BTCUSD",
        "side": "Buy",
        "qty": 1,
        "price": 50000,
        "status": "Filled"
    }
    save_trade_record("BTCUSD", "Buy", 1, 50000, "Market", "Filled", test_order)

    # 투자 성과 저장 테스트
    save_investment_performance(10, 7, 70.0)
