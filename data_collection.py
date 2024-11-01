# data_collection.py

import requests
import pandas as pd
from bybit import bybit
import os
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Bybit 클라이언트 설정
client = bybit(test=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

def get_market_data(symbol="BTCUSD", interval="1", limit=200):
    """
    Bybit에서 시장 데이터(캔들스틱 데이터)를 수집합니다.
    """
    try:
        response = client.Kline.Kline_get(symbol=symbol, interval=interval, limit=limit).result()
        data = response[0]['result']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"시장 데이터 수집 에러: {e}")
        return pd.DataFrame()

def get_order_history(symbol="BTCUSD", limit=100):
    """
    Bybit에서 거래 내역을 수집합니다.
    """
    try:
        response = client.Order.Order_getOrders(symbol=symbol, limit=limit).result()
        orders = response[0]['result']
        return orders
    except Exception as e:
        print(f"거래 내역 수집 에러: {e}")
        return []

def get_fear_greed_index():
    """
    공포 탐욕 지수를 수집합니다. API가 없을 경우 웹 스크래핑을 사용.
    """
    try:
        url = "https://alternative.me/crypto/fear-and-greed-index/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        index_element = soup.find("div", {"class": "fng-circle"})
        index = index_element.text.strip() if index_element else "N/A"
        return index
    except Exception as e:
        print(f"공포 탐욕 지수 수집 에러: {e}")
        return "N/A"

# 테스트용 호출
if __name__ == "__main__":
    df = get_market_data()
    print(df.head())

    orders = get_order_history()
    print(json.dumps(orders, indent=2))

    fg_index = get_fear_greed_index()
    print(f"공포 탐욕 지수: {fg_index}")
