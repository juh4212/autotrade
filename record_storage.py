# record_storage.py

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

# 데이터베이스 설정
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class TradeRecord(Base):
    __tablename__ = 'trade_records'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    side = Column(String)
    qty = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    order_type = Column(String)
    status = Column(String)
    response = Column(Text)

class InvestmentPerformance(Base):
    __tablename__ = 'investment_performance'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_trades = Column(Integer)
    profitable_trades = Column(Integer)
    win_rate = Column(Float)

# 테이블 생성
Base.metadata.create_all(engine)

def save_trade_record(symbol, side, qty, price, order_type, status, response):
    """
    거래 내역을 데이터베이스에 저장합니다.
    """
    try:
        record = TradeRecord(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_type=order_type,
            status=status,
            response=json.dumps(response)
        )
        session.add(record)
        session.commit()
    except Exception as e:
        print(f"거래 기록 저장 에러: {e}")
        session.rollback()

def save_investment_performance(total_trades, profitable_trades, win_rate):
    """
    투자 성과를 데이터베이스에 저장합니다.
    """
    try:
        performance = InvestmentPerformance(
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            win_rate=win_rate
        )
        session.add(performance)
        session.commit()
    except Exception as e:
        print(f"투자 성과 저장 에러: {e}")
        session.rollback()

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

