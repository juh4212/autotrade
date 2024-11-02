# data_collection.py

import requests
import logging
from pybit.unified_trading import HTTP
import os
import pandas as pd

# Bybit 클라이언트 초기화
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

if BYBIT_API_KEY and BYBIT_API_SECRET:
    bybit_client = HTTP(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET
    )
    logging.info("Bybit 클라이언트가 초기화되었습니다.")
else:
    bybit_client = None
    logging.error("Bybit API 키 또는 시크릿이 설정되지 않았습니다.")

def get_fear_greed_index():
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=1')
        if response.status_code == 200:
            data = response.json()
            return data['data'][0]['value']
        else:
            logging.error(f"공포 탐욕 지수 가져오기 실패: HTTP {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"공포 탐욕 지수 가져오기 중 에러 발생: {e}")
        return None

def get_market_data():
    # 시장 데이터 수집 로직 구현
    # 예시로 빈 DataFrame 반환
    return pd.DataFrame()

def get_order_history(symbol, limit=100):
    if bybit_client:
        try:
            response = bybit_client.get_order_history(symbol=symbol, limit=limit)
            return response.get('result', [])
        except Exception as e:
            logging.error(f"주문 이력 가져오기 중 에러 발생: {e}")
            return []
    else:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return []

def get_wallet_balance():
    if bybit_client:
        try:
            # 잔고 정보 가져오기 (accountType='CONTRACT')
            response = bybit_client.get_wallet_balance(coin="USDT", accountType="CONTRACT")
            logging.debug(f"get_wallet_balance response: {response}")  # 디버그용 로그 추가

            if response.get("retCode") == 0:
                balances = response["result"]["balances"]
                usdt_balance = None
                for balance in balances:
                    if balance["coin"] == "USDT":
                        usdt_balance = balance["available"]
                        break
                if usdt_balance is not None:
                    return usdt_balance
                else:
                    logging.error("USDT 잔고 정보를 찾을 수 없습니다.")
                    return None
            else:
                logging.error(f"잔고 정보 가져오기 실패: {response.get('retMsg')}")
                return None
        except Exception as e:
            logging.error(f"잔고 정보 가져오기 중 에러 발생: {e}")
            return None
    else:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None
