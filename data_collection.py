# data_collection.py

import logging
import requests
from pybit.unified_trading import HTTP
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Bybit 클라이언트 초기화
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('USE_TESTNET', 'False').lower() in ['true', '1', 't']

if BYBIT_API_KEY and BYBIT_API_SECRET:
    try:
        bybit_client = HTTP(
            testnet=USE_TESTNET,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        logging.info("Bybit 클라이언트가 초기화되었습니다.")
    except Exception as e:
        bybit_client = None
        logging.error(f"Bybit 클라이언트 초기화 실패: {e}")
else:
    bybit_client = None
    logging.error("Bybit API 키 또는 시크릿이 설정되지 않았습니다.")

def get_wallet_balance():
    """
    Bybit 지갑 잔고 정보를 가져오는 함수
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_wallet_balance(coin='USDT')
        if response['retCode'] == 0:
            balance_info = response['result']['list'][0]
            equity = float(balance_info['equity'])
            available_balance = float(balance_info['availableBalance'])
            logging.info(f"총 자산 (Equity): {equity} USDT")
            logging.info(f"사용 가능 잔액: {available_balance} USDT")
            return {
                "equity": equity,
                "available_balance": available_balance
            }
        else:
            logging.error(f"잔고 정보를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"잔고 정보를 가져오는 중 예외 발생: {e}")
        return None

def get_market_data(symbol):
    """
    Bybit에서 지정된 심볼의 시장 데이터를 가져옵니다.
    
    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
    
    Returns:
        dict: 시장 데이터
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_orderbook(
            category='linear',
            symbol=symbol,
            limit=50
        )
        if response['retCode'] == 0:
            orderbook = response['result']
            return orderbook
        else:
            logging.error(f"시장 데이터를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"시장 데이터를 가져오는 중 예외 발생: {e}")
        return None

# 필요에 따라 추가적인 데이터 수집 함수들을 구현합니다.
