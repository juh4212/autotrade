import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
import openai
import ta
from ta.utils import dropna
import time
import requests
import logging
from pymongo import MongoClient
from datetime import datetime, timedelta
import re
import schedule
import numpy as np

# .env 파일에 저장된 환경 변수를 불러오기 (API 키 등)
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upbit 객체 생성
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
if not access or not secret:
    logger.error("API 키를 찾을 수 없습니다. .env 파일을 확인해주세요.")
    raise ValueError("API 키가 없습니다. .env 파일을 확인해주세요.")
upbit = pyupbit.Upbit(access, secret)

# MongoDB 데이터베이스 초기화 함수 - 거래 내역을 저장할 컬렉션을 생성
def init_db():
    client = MongoClient('mongodb://localhost:27017/')  # MongoDB 연결
    db = client['bitcoin_trades_db']  # 데이터베이스 선택
    trades_collection = db['trades']  # 컬렉션 선택
    return trades_collection

# 거래 기록을 DB에 저장하는 함수
def log_trade(trades_collection, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    trade_data = {
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "percentage": percentage,
        "reason": reason,
        "btc_balance": btc_balance,
        "krw_balance": krw_balance,
        "btc_avg_buy_price": btc_avg_buy_price,
        "btc_krw_price": btc_krw_price,
        "reflection": reflection
    }
    trades_collection.insert_one(trade_data)

# 최근 투자 기록 조회
def get_recent_trades(trades_collection, days=7):
    seven_days_ago = datetime.now() - timedelta(days=days)
    recent_trades_cursor = trades_collection.find({
        "timestamp": {"$gte": seven_days_ago.isoformat()}
    }).sort("timestamp", -1)
    trades_df = pd.DataFrame(list(recent_trades_cursor))
    if '_id' in trades_df.columns:
        trades_df.drop(columns=['_id'], inplace=True)  # MongoDB의 기본 '_id' 필드 제거
    return trades_df

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (KRW + BTC * 당시 가격)
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    # 최종 잔고 계산
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    return (final_balance - initial_balance) / initial_balance * 100

# OpenAI API 호출 함수 (타임아웃 및 재시도 로직 포함)
def get_openai_response(messages):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API 키가 없거나 유효하지 않습니다.")
        return None

    retries = 3  # 최대 재시도 횟수
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                timeout=60  # 타임아웃 설정 (초)
            )
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
            time.sleep(2 ** i)  # 지수 백오프 (재시도 전 대기)
    return None

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    messages = [
        {
            "role": "user",
            "content": "당신은 최근 거래 성과와 현재 시장 상황을 분석하여 향후 거래 결정을 위한 인사이트와 개선 사항을 생성하는 AI 트레이딩 어시스턴트입니다."
        },
        {
            "role": "user",
            "content": f"""
최근 거래 데이터:
{trades_df.to_json(orient='records')}

현재 시장 데이터:
{current_market_data}

지난 7일 동안의 전체 퍼포먼스: {performance:.2f}%

이 데이터를 분석하고 다음을 제공해주세요:
1. 최근 거래 결정에 대한 간단한 반성
2. 잘된 점과 개선이 필요한 점에 대한 인사이트
3. 향후 거래 결정을 위한 개선 제안
4. 시장 데이터에서 발견한 패턴이나 트렌드

응답은 250자 이내로 제한해주세요.
"""
        }
    ]

    return get_openai_response(messages)

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df):
    # 볼린저 밴드 추가
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI (Relative Strength Index) 추가
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence) 추가
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 이동평균선 (단기, 장기)
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()

    # Stochastic Oscillator 추가
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Average True Range (ATR) 추가
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # On-Balance Volume (OBV) 추가
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['close'], volume=df['volume']).on_balance_volume()

    return df

### 메인 AI 트레이딩 로직
def ai_trading():
    global upbit
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    all_balances = upbit.get_balances()
    filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
    
    # 2. 오더북(호가 데이터) 조회
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    
    # 3. 차트 데이터 조회 및 보조지표 추가
    df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=180)
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)
    
    df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=168)  # 7일간의 시간별 데이터
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)

    # 최근 데이터만 사용하도록 설정 (메모리 절약)
    df_daily_recent = df_daily.tail(60)
    df_hourly_recent = df_hourly.tail(48)
    
    ### AI에게 데이터 제공하고 판단 받기
    logger.info("AI 분석을 위한 데이터 준비 중...")
    trades_collection = init_db()
    recent_trades = get_recent_trades(trades_collection)
    current_market_data = {
        "orderbook": orderbook,
        "daily_ohlcv": df_daily_recent.to_dict(),
        "hourly_ohlcv": df_hourly_recent.to_dict()
    }
    reflection = generate_reflection(recent_trades, current_market_data)

    if reflection is None:
        logger.error("AI로부터 유효한 응답을 받지 못했습니다. 트레이딩 작업을 중단합니다.")
        return

    logger.info(f"AI 반성 내용: {reflection}")

    # 거래 결정 예시 로직 (추가적인 API 호출과 거래 실행 로직 포함 가능)
    # 실제로 실행하기 전에 충분한 테스트와 검토가 필요합니다.

if __name__ == "__main__":
    # MongoDB 컬렉션 초기화
    trades_collection = init_db()

    # 중복 실행 방지를 위한 변수
    trading_in_progress = False

    # 트레이딩 작업을 수행하는 함수
    def job():
        global trading_in_progress
        if trading_in_progress:
            logger.warning("트레이딩 작업이 이미 진행 중입니다. 이번 실행은 건너뜁니다.")
            return
        try:
            trading_in_progress = True
            ai_trading()
        except Exception as e:
            logger.error(f"오류 발생: {e}")
        finally:
            trading_in_progress = False

    # 테스트
    # job()

    # 매 4시간마다 실행
    schedule.every().day.at("00:00").do(job)
    schedule.every().day.at("04:00").do(job)
    schedule.every().day.at("08:00").do(job)
    schedule.every().day.at("12:00").do(job)
    schedule.every().day.at("16:00").do(job)
    schedule.every().day.at("20:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
