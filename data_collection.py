# data_collection.py

import logging
import pyupbit  # pyupbit 라이브러리 사용 가정

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 필요 시 DEBUG로 변경
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_recent_trades(conn):
    """
    데이터베이스에서 최근 거래 내역을 가져오는 함수
    """
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100", conn)
        return df
    except Exception as e:
        logging.error(f"최근 거래 내역 조회 실패: {e}")
        return pd.DataFrame()

def get_current_market_data():
    """
    현재 시장 데이터를 가져오는 함수
    """
    try:
        # 예시: 현재 가격, 오더북, 뉴스 헤드라인 등
        current_price = pyupbit.get_current_price("KRW-BTC")
        orderbook = pyupbit.get_orderbook("KRW-BTC")
        # 뉴스 헤드라인 및 공포 탐욕 지수는 별도의 API 호출 필요
        # 여기서는 예시 데이터로 대체
        news_headlines = ["Bitcoin hits new high", "Market sentiment is bullish"]
        fear_greed_index = 60  # 예시 값
        daily_ohlcv = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=60)
        hourly_ohlcv = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=48)
        return {
            "current_price": current_price,
            "orderbook": orderbook,
            "news_headlines": news_headlines,
            "fear_greed_index": fear_greed_index,
            "daily_ohlcv": daily_ohlcv,
            "hourly_ohlcv": hourly_ohlcv
        }
    except Exception as e:
        logging.error(f"현재 시장 데이터 조회 실패: {e}")
        return {}

def get_usdt_balance():
    """
    USDT 잔고를 조회하는 함수
    """
    try:
        # 예시: Bybit 클라이언트에서 잔고 조회
        balance_info = bybit_client.get_wallet_balance()
        usdt_balance = balance_info.get("USDT", {}).get("available_balance", 0)
        return float(usdt_balance)
    except Exception as e:
        logging.error(f"USDT 잔고 조회 실패: {e}")
        return None

def get_btc_balance():
    """
    BTC 잔고를 조회하는 함수
    """
    try:
        balance_info = bybit_client.get_wallet_balance()
        btc_balance = balance_info.get("BTC", {}).get("available_balance", 0)
        return float(btc_balance)
    except Exception as e:
        logging.error(f"BTC 잔고 조회 실패: {e}")
        return None

def get_current_price(symbol):
    """
    특정 심볼의 현재 가격을 조회하는 함수
    """
    try:
        price = pyupbit.get_current_price(symbol)
        return price
    except Exception as e:
        logging.error(f"{symbol}의 현재 가격 조회 실패: {e}")
        return None
