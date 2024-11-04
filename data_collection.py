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

# logger 객체 생성
logger = logging.getLogger(__name__)

# 환경 변수에서 Bybit API 키 및 시크릿 가져오기
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('USE_TESTNET', 'False').lower() in ['true', '1', 't']

# Bybit 클라이언트 초기화
if BYBIT_API_KEY and BYBIT_API_SECRET:
    try:
        bybit_client = HTTP(
            testnet=USE_TESTNET,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        logger.info("Bybit 클라이언트가 초기화되었습니다.")
    except Exception as e:
        bybit_client = None
        logger.error(f"Bybit 클라이언트 초기화 실패: {e}")
else:
    bybit_client = None
    logger.error("Bybit API 키 또는 시크릿이 설정되지 않았습니다.")

def get_account_balance(bybit, account_type='CONTRACT', coin='USDT'):
    """
    Bybit 지갑 잔고 정보를 가져오는 함수

    Parameters:
        bybit (HTTP): pybit.unified_trading.HTTP 클라이언트 객체
        account_type (str): 'CONTRACT' 또는 'SPOT'
        coin (str, optional): 조회할 코인 (예: 'USDT'). CONTRACT 계정일 때 사용.

    Returns:
        dict: 잔고 정보 {'equity': float, 'available_balance': float} 또는 None
    """
    try:
        logger.debug(f"get_account_balance 호출: account_type={account_type}, coin={coin}")
        logger.debug(f"account_type type: {type(account_type)}, coin type: {type(coin)}")

        # API 호출
        if account_type.upper() == 'CONTRACT':
            response = bybit.get_wallet_balance(accountType=account_type.upper(), coin=coin.upper())
        elif account_type.upper() == 'SPOT':
            response = bybit.get_wallet_balance(accountType=account_type.upper())
        else:
            logger.error(f"올바르지 않은 account_type: {account_type}")
            return None

        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logger.debug(f"Bybit API 응답 데이터: {response_json}")  # 전체 응답 데이터 출력

        # 응답 확인
        if response.get('retCode') == 0 and 'result' in response:
            account_list = response['result'].get('list', [])
            if account_list:
                account_info = account_list[0]
                logger.debug(f"Account Info: {account_info}")

                if account_type.upper() == 'CONTRACT':
                    # CONTRACT 계정의 경우 'coin' 필드가 리스트인지 확인
                    coin_balances = account_info.get('coin', [])
                    logger.debug(f"Coin Balances: {coin_balances}")

                    if isinstance(coin_balances, list):
                        usdt_balance = next((coin for coin in coin_balances if coin.get('coin') == 'USDT'), None)
                        if usdt_balance:
                            # 모든 필드를 로그로 출력
                            for key, value in usdt_balance.items():
                                logger.debug(f"USDT Balance Field - {key}: {value}")

                            equity = float(usdt_balance.get('equity', 0))
                            available_balance = float(usdt_balance.get('availableBalance', 0))  # 'availableBalance' 사용
                            
                            logger.info(f"USDT 전체 자산: {equity}, 사용 가능한 자산: {available_balance}")
                            return {
                                "equity": equity,
                                "available_balance": available_balance
                            }
                        else:
                            logger.error("USDT 잔고 데이터를 찾을 수 없습니다.")
                            return None
                    else:
                        logger.error(f"'coin' 필드의 타입이 예상과 다릅니다: {type(coin_balances)}")
                        return None
                elif account_type.upper() == 'SPOT':
                    # SPOT 계정의 경우 모든 자산의 잔고를 합산
                    spot_balances = account_info.get('list', [])
                    logger.debug(f"Spot Balances: {spot_balances}")

                    if isinstance(spot_balances, list):
                        total_equity = 0.0
                        total_available_balance = 0.0
                        for asset in spot_balances:
                            equity = float(asset.get('equity', 0))
                            available_balance = float(asset.get('availableBalance', 0))
                            total_equity += equity
                            total_available_balance += available_balance
                        logger.info(f"SPOT 계정 총 자산 (Equity): {total_equity} USDT")
                        logger.info(f"SPOT 계정 사용 가능 잔액: {total_available_balance} USDT")
                        return {
                            "equity": total_equity,
                            "available_balance": total_available_balance
                        }
                    else:
                        logger.error(f"'list' 필드의 타입이 예상과 다릅니다: {type(spot_balances)}")
                        return None
                else:
                    logger.error("올바르지 않은 계정 유형 또는 파라미터.")
                    return None
            else:
                logger.error("계정 리스트가 비어 있습니다.")
                return None
        else:
            logger.error(f"잔고 데이터를 가져오지 못했습니다. retCode: {response.get('retCode')}, retMsg: {response.get('retMsg')}")
            return None
    except Exception as e:
        logger.error(f"Bybit 잔고 조회 오류: {e}")
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
        logger.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_orderbook(
            category='linear',
            symbol=symbol.upper()
        )
        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logger.debug(f"get_market_data 응답 데이터: {response_json}")  # 전체 응답 데이터 출력

        if response.get('retCode') == 0:
            orderbook = response['result']
            logger.info(f"{symbol.upper()}의 시장 데이터를 가져왔습니다.")
            return orderbook
        else:
            logger.error(f"시장 데이터를 가져오는 중 에러 발생: {response.get('retMsg')}")
            return None
    except Exception as e:
        logger.error(f"시장 데이터를 가져오는 중 예외 발생: {e}")
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
        logger.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        response = bybit_client.get_public_trading_records(
            category='linear',
            symbol=symbol.upper(),
            limit=limit
        )
        # 응답을 JSON 문자열로 포맷팅하여 로그에 기록
        response_json = json.dumps(response, indent=4, ensure_ascii=False)
        logger.debug(f"get_recent_trades 응답 데이터: {response_json}")  # 전체 응답 데이터 출력

        if response.get('retCode') == 0:
            trades = response['result']['list']
            logger.info(f"{symbol.upper()}의 최근 거래 내역을 가져왔습니다.")
            return trades
        else:
            logger.error(f"최근 거래 내역을 가져오는 중 에러 발생: {response.get('retMsg')}")
            return None
    except Exception as e:
        logger.error(f"최근 거래 내역을 가져오는 중 예외 발생: {e}")
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
        logger.error("Bybit 클라이언트가 초기화되지 않았습니다.")
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
        logger.debug(f"get_kline_data 응답 데이터: {response_json}")  # 전체 응답 데이터 출력

        if response.get('retCode') == 0:
            kline_data = response['result']['list']
            logger.info(f"{symbol.upper()}의 캔들 차트 데이터를 가져왔습니다.")
            return kline_data
        else:
            logger.error(f"캔들 차트 데이터를 가져오는 중 에러 발생: {response.get('retMsg')}")
            return None
    except Exception as e:
        logger.error(f"캔들 차트 데이터를 가져오는 중 예외 발생: {e}")
        return None

# 필요에 따라 추가적인 데이터 수집 함수들을 구현합니다.
