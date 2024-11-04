# data_collection.py

import logging
import os
import json
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

def get_wallet_balance(account_type='CONTRACT', coin='USDT'):
    """
    Bybit 지갑 잔고 정보를 가져오는 함수

    Parameters:
        account_type (str): 'CONTRACT' 또는 'SPOT'
        coin (str, optional): 조회할 코인 (예: 'USDT'). CONTRACT 계정일 때 사용.

    Returns:
        dict: 잔고 정보 {'equity': float, 'available_balance': float} 또는 None
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        logging.debug(f"get_wallet_balance 호출: account_type={account_type}, coin={coin}")
        logging.debug(f"account_type type: {type(account_type)}, coin type: {type(coin)}")

        if isinstance(account_type, str):
            account_type_str = account_type.upper()
        else:
            logging.error(f"account_type이 문자열이 아닙니다: {account_type}")
            return None

        if isinstance(coin, str):
            coin_str = coin.upper()
        else:
            logging.error(f"coin이 문자열이 아닙니다: {coin}")
            return None

        if account_type_str == 'CONTRACT' and coin_str:
            response = bybit_client.get_wallet_balance(
                accountType='CONTRACT',
                coin=coin_str
            )
        elif account_type_str == 'SPOT':
            response = bybit_client.get_wallet_balance(
                accountType='SPOT'
            )
        else:
            logging.error("올바르지 않은 계정 유형 또는 파라미터.")
            return None

        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logging.debug(f"get_wallet_balance 응답: {response_json}")  # 응답 전체 로그에 기록

        if response['retCode'] == 0:
            if account_type_str == 'CONTRACT' and coin_str:
                # CONTRACT 계정의 경우 특정 코인의 잔고 정보 사용
                if 'list' in response['result'] and isinstance(response['result']['list'], list):
                    if len(response['result']['list']) > 0:
                        coin_info = response['result']['list'][0]
                        logging.debug(f"coin_info: {coin_info}")
                        logging.debug(f"coin_info type: {type(coin_info)}")
                        if isinstance(coin_info, dict) and 'coin' in coin_info and isinstance(coin_info['coin'], str) and coin_info['coin'].upper() == coin_str:
                            equity = float(coin_info.get('equity', '0'))
                            available_balance = float(coin_info.get('availableBalance', '0'))
                            logging.info(f"총 자산 (Equity): {equity} {coin_str}")
                            logging.info(f"사용 가능 잔액: {available_balance} {coin_str}")
                            return {
                                "equity": equity,
                                "available_balance": available_balance
                            }
                        else:
                            logging.error(f"{coin_str} 잔고 정보가 없습니다.")
                            logging.debug(f"coin_info 내용: {coin_info}")
                            return None
                    else:
                        logging.error("잔고 리스트가 비어 있습니다.")
                        return None
                else:
                    logging.error("응답 결과에서 'list' 키가 없거나 리스트가 아닙니다.")
                    return None
            elif account_type_str == 'SPOT':
                # SPOT 계정의 경우 전체 잔고 조회
                if 'list' in response['result'] and isinstance(response['result']['list'], list):
                    account_info = response['result']['list']
                    logging.debug(f"account_info: {account_info}")
                    logging.debug(f"account_info type: {type(account_info)}, first item type: {type(account_info[0]) if len(account_info) > 0 else 'N/A'}")
                    total_equity = 0.0
                    total_available_balance = 0.0
                    for asset in account_info:
                        asset_equity = float(asset.get('equity', '0'))
                        asset_available = float(asset.get('availableBalance', '0'))
                        total_equity += asset_equity
                        total_available_balance += asset_available
                    logging.info(f"SPOT 계정 총 자산 (Equity): {total_equity} USDT")
                    logging.info(f"SPOT 계정 사용 가능 잔액: {total_available_balance} USDT")
                    return {
                        "equity": total_equity,
                        "available_balance": total_available_balance
                    }
                else:
                    logging.error("응답 결과에서 'list' 키가 없거나 리스트가 아닙니다.")
                    return None
            else:
                logging.error("올바르지 않은 계정 유형 또는 파라미터.")
                return None
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
            symbol=symbol.upper()
        )
        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logging.debug(f"get_market_data 응답: {response_json}")  # 응답 전체 로그에 기록

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
        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logging.debug(f"get_recent_trades 응답: {response_json}")  # 응답 전체 로그에 기록

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
        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logging.debug(f"get_kline_data 응답: {response_json}")  # 응답 전체 로그에 기록

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
