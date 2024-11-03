# data_collection.py

import logging
import os
from pybit.unified_trading import HTTP

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # 디버깅을 위해 DEBUG 레벨로 설정
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
        response = bybit_client.get_wallet_balance(
            accountType='UNIFIED',  # 'UNIFIED' 또는 'CONTRACT'로 설정
            coin='USDT'             # 'USDT' 잔고 조회
        )
        logging.debug(f"get_wallet_balance 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            # UNIFIED 계정의 경우 'totalAvailableBalance'를 사용
            total_available_balance = float(response['result']['list'][0].get('totalAvailableBalance', '0'))
            total_equity = float(response['result']['list'][0].get('totalEquity', '0'))
            logging.info(f"총 자산 (Equity): {total_equity} USDT")
            logging.info(f"사용 가능 잔액: {total_available_balance} USDT")
            return {
                "equity": total_equity,
                "available_balance": total_available_balance
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
            symbol=symbol
        )
        logging.debug(f"get_market_data 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            orderbook = response['result']
            logging.info(f"{symbol}의 시장 데이터를 가져왔습니다.")
            return orderbook
        else:
            logging.error(f"시장 데이터를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"시장 데이터를 가져오는 중 예외 발생: {e}")
        return None

def get_recent_trades(symbol, limit=50):
    """
    Bybit에서 지정된 심볼의 최근 거래 내역을 가져옵니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        limit (int): 가져올 거래 수 (기본값: 50)

    Returns:
        list: 최근 거래 내역
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_public_trading_records(
            category='linear',
            symbol=symbol,
            limit=limit
        )
        logging.debug(f"get_recent_trades 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            trades = response['result']['list']
            logging.info(f"{symbol}의 최근 거래 내역을 가져왔습니다.")
            return trades
        else:
            logging.error(f"최근 거래 내역을 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"최근 거래 내역을 가져오는 중 예외 발생: {e}")
        return None

def get_kline_data(symbol, interval='15', limit=200):
    """
    Bybit에서 지정된 심볼의 캔들 차트 데이터를 가져옵니다.

    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        interval (str): 캔들 차트 간격 (예: '1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
        limit (int): 가져올 데이터 수 (기본값: 200)

    Returns:
        list: 캔들 차트 데이터
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_kline(
            category='linear',
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        logging.debug(f"get_kline_data 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            kline_data = response['result']['list']
            logging.info(f"{symbol}의 캔들 차트 데이터를 가져왔습니다.")
            return kline_data
        else:
            logging.error(f"캔들 차트 데이터를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"캔들 차트 데이터를 가져오는 중 예외 발생: {e}")
        return None

# 필요에 따라 추가적인 데이터 수집 함수들을 구현합니다.
