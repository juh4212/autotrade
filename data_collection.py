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

def get_account_info():
    """
    Bybit 계정 정보를 가져오는 함수

    Returns:
        dict: 계정 정보 또는 None
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_wallet_balance()  # 기본적으로 UNIFIED 계정 정보 조회
        logging.debug(f"get_account_info 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            account_info = response['result']['list'][0]
            unified_margin_status = int(account_info.get('unifiedMarginStatus', '1'))  # 기본값: classic
            logging.info(f"unifiedMarginStatus: {unified_margin_status}")
            return account_info
        else:
            logging.error(f"계정 정보를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"계정 정보를 가져오는 중 예외 발생: {e}")
        return None

def determine_account_mode(unified_margin_status):
    """
    unifiedMarginStatus 값을 기반으로 계정 유형을 결정하는 함수

    Parameters:
        unified_margin_status (int): unifiedMarginStatus 값

    Returns:
        str: 계정 유형 ('classic', 'uta1.0', 'uta2.0') 또는 None
    """
    if unified_margin_status == 1:
        return 'classic'
    elif unified_margin_status in [3, 4]:
        return 'uta1.0'
    elif unified_margin_status in [5, 6]:
        return 'uta2.0'
    else:
        logging.error(f"알 수 없는 unifiedMarginStatus 값: {unified_margin_status}")
        return None

def get_unified_wallet_balance(account_info):
    """
    UTA 2.0 계정의 USDT 잔고 정보를 가져오는 함수

    Parameters:
        account_info (dict): 계정 정보

    Returns:
        dict: 잔고 정보 {'equity': float, 'available_balance': float} 또는 None
    """
    try:
        total_equity = float(account_info.get('totalEquity', '0'))
        total_available_balance = float(account_info.get('totalAvailableBalance', '0'))
        logging.info(f"총 자산 (Equity): {total_equity} USDT")
        logging.info(f"사용 가능 잔액: {total_available_balance} USDT")
        return {
            "equity": total_equity,
            "available_balance": total_available_balance
        }
    except Exception as e:
        logging.error(f"UTA 2.0 잔고 정보를 파싱하는 중 예외 발생: {e}")
        return None

def get_contract_wallet_balance(coin='USDT'):
    """
    CONTRACT 계정의 특정 코인 잔고 정보를 가져오는 함수

    Parameters:
        coin (str): 조회할 코인 (기본값: 'USDT')

    Returns:
        dict: 잔고 정보 {'equity': float, 'available_balance': float} 또는 None
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_wallet_balance(
            accountType='CONTRACT',
            coin=coin.upper()
        )
        logging.debug(f"get_contract_wallet_balance 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            coin_info = response['result']['list'][0]['coin']
            if coin_info and coin_info['coin'].upper() == coin.upper():
                equity = float(coin_info.get('equity', '0'))
                available_balance = float(coin_info.get('availableBalance', '0'))
                logging.info(f"총 자산 (Equity): {equity} {coin.upper()}")
                logging.info(f"사용 가능 잔액: {available_balance} {coin.upper()}")
                return {
                    "equity": equity,
                    "available_balance": available_balance
                }
            else:
                logging.error(f"{coin.upper()} 잔고 정보가 없습니다.")
                return None
        else:
            logging.error(f"CONTRACT 잔고 정보를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"CONTRACT 잔고 정보를 가져오는 중 예외 발생: {e}")
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
            symbol=symbol.upper()
        )
        logging.debug(f"get_market_data 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            orderbook = response['result']
            logging.info(f"{symbol.upper()}의 시장 데이터를 가져왔습니다.")
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
            symbol=symbol.upper(),
            limit=limit
        )
        logging.debug(f"get_recent_trades 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            trades = response['result']['list']
            logging.info(f"{symbol.upper()}의 최근 거래 내역을 가져왔습니다.")
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
            symbol=symbol.upper(),
            interval=interval,
            limit=limit
        )
        logging.debug(f"get_kline_data 응답: {response}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            kline_data = response['result']['list']
            logging.info(f"{symbol.upper()}의 캔들 차트 데이터를 가져왔습니다.")
            return kline_data
        else:
            logging.error(f"캔들 차트 데이터를 가져오는 중 에러 발생: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"캔들 차트 데이터를 가져오는 중 예외 발생: {e}")
        return None

# 필요에 따라 추가적인 데이터 수집 함수들을 구현합니다.
