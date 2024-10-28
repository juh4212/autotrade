import os
from dotenv import load_dotenv
from pybit.usdt_perpetual import HTTP  # 올바른 임포트 경로 사용
import pandas as pd
import json
import openai
import ta
from ta.utils import dropna
import time
import requests
import logging
from datetime import datetime, timedelta
import re
import schedule
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# .env 파일에 저장된 환경 변수를 불러오기 (API 키 등)
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bybit 객체 생성
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
if not api_key or not api_secret:
    logger.error("API keys not found. Please check your .env file.")
    raise ValueError("Missing API keys. Please check your .env file.")

# Bybit의 REST API 엔드포인트 설정
session = HTTP("https://api.bybit.com", api_key=api_key, api_secret=api_secret)

# MongoDB 초기화 함수 - 거래 내역을 저장할 컬렉션을 설정
def init_db():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        logger.error("MongoDB URI not found. Please check your .env file.")
        raise ValueError("Missing MongoDB URI. Please check your .env file.")
    try:
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']  # 데이터베이스 이름
        trades_collection = db['trades']  # 컬렉션 이름
        # 인덱스 생성 (timestamp)
        trades_collection.create_index("timestamp")
        return trades_collection
    except PyMongoError as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

# 거래 기록을 DB에 저장하는 함수
def log_trade(trades_collection, trade_data):
    try:
        trades_collection.insert_one(trade_data)
        logger.info("Trade logged successfully.")
    except PyMongoError as e:
        logger.error(f"Failed to log trade: {e}")

# 최근 투자 기록 조회
def get_recent_trades(trades_collection, days=7):
    seven_days_ago = datetime.utcnow() - timedelta(days=days)
    try:
        cursor = trades_collection.find({"timestamp": {"$gte": seven_days_ago}}).sort("timestamp", -1)
        trades = list(cursor)
        return pd.DataFrame(trades)
    except PyMongoError as e:
        logger.error(f"Failed to retrieve recent trades: {e}")
        return pd.DataFrame()

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (USD + BTC * 현재 가격)
    initial_balance = trades_df.iloc[-1]['usd_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_usd_price']
    # 최종 잔고 계산
    final_balance = trades_df.iloc[0]['usd_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_usd_price']
    return (final_balance - initial_balance) / initial_balance * 100

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."
                },
                {
                    "role": "user",
                    "content": f"""
Recent trading data:
{trades_df.to_json(orient='records')}

Current market data:
{json.dumps(current_market_data)}

Overall performance in the last 7 days: {performance:.2f}%

Please analyze this data and provide:
1. A brief reflection on the recent trading decisions
2. Insights on what worked well and what didn't
3. Suggestions for improvement in future trading decisions
4. Any patterns or trends you notice in the market data

Limit your response to 250 words or less.
"""
                }
            ]
        )
        response_content = response['choices'][0]['message']['content']
        return response_content
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        return None

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

# 공포 탐욕 지수 조회
def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")
        return None

### 메인 AI 트레이딩 로직
def ai_trading(trades_collection):
    global session
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    try:
        wallet = session.get_wallet_balance()
        if 'result' not in wallet or not wallet['result']:
            logger.error("Failed to retrieve wallet balance.")
            return
        balances = wallet['result']
        btc_balance = float(next((item['coin_balance'] for item in balances if item['coin'] == 'BTC'), 0))
        usd_balance = float(next((item['coin_balance'] for item in balances if item['coin'] == 'USD'), 0))
    except Exception as e:
        logger.error(f"Error fetching balances: {e}")
        return

    # 2. 오더북(호가 데이터) 조회
    try:
        orderbook = session.order_book(symbol="BTCUSD")
    except Exception as e:
        logger.error(f"Error fetching order book: {e}")
        orderbook = {}

    # 3. 차트 데이터 조회 및 보조지표 추가
    try:
        # Bybit에서는 캔들 데이터 요청 시 timeframe을 설정해야 합니다.
        df_daily = pd.DataFrame(session.query_kline(symbol="BTCUSD", interval="D", limit=180)['result'])
        df_daily = dropna(df_daily)
        df_daily = add_indicators(df_daily)
        
        df_hourly = pd.DataFrame(session.query_kline(symbol="BTCUSD", interval="60", limit=168)['result'])  # 7 days of hourly data
        df_hourly = dropna(df_hourly)
        df_hourly = add_indicators(df_hourly)
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        df_daily = pd.DataFrame()
        df_hourly = pd.DataFrame()

    # 최근 데이터만 사용하도록 설정 (메모리 절약)
    df_daily_recent = df_daily.tail(60) if not df_daily.empty else pd.DataFrame()
    df_hourly_recent = df_hourly.tail(48) if not df_hourly.empty else pd.DataFrame()

    # 4. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()

    ### AI에게 데이터 제공하고 판단 받기
    try:
        # 최근 거래 내역 가져오기
        recent_trades = get_recent_trades(trades_collection)
        
        # 현재 시장 데이터 수집 (news_headlines 제거)
        current_market_data = {
            "fear_greed_index": fear_greed_index,
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.to_dict(orient='records'),
            "hourly_ohlcv": df_hourly_recent.to_dict(orient='records')
        }
        
        # 반성 및 개선 내용 생성
        reflection = generate_reflection(recent_trades, current_market_data)
        
        if reflection is None:
            logger.error("Failed to generate reflection.")
            return
        
        # AI 모델에 반성 내용 제공
        # Few-shot prompting으로 JSON 예시 추가
        examples = """
Example Response 1:
{
  "decision": "buy",
  "percentage": 50,
  "reason": "Based on the current market indicators and positive trends, it's a good opportunity to invest."
}

Example Response 2:
{
  "decision": "sell",
  "percentage": 30,
  "reason": "Due to negative trends in the market and high fear index, it is advisable to reduce holdings."
}

Example Response 3:
{
  "decision": "hold",
  "percentage": 0,
  "reason": "Market indicators are neutral; it's best to wait for a clearer signal."
}
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Bitcoin investing."
                },
                {
                    "role": "user",
                    "content": f"""You are an expert in Bitcoin investing. This analysis is performed every 4 hours. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:

- Technical indicators and market data
- The Fear and Greed Index and its implications
- Overall market sentiment
- Recent trading performance and reflection

Recent trading reflection:
{reflection}

Based on your analysis, make a decision and provide your reasoning.

Please provide your response in the following JSON format:

{examples}

Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
Your percentage should reflect the strength of your conviction in the decision based on the analyzed data.
"""
                },
                {
                    "role": "user",
                    "content": f"""Current investment status: {{'BTC': {btc_balance}, 'USD': {usd_balance}}}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json(orient='records')}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json(orient='records')}
Fear and Greed Index: {json.dumps(fear_greed_index)}
"""
                }
            ]
        )

        response_text = response['choices'][0]['message']['content']

        # AI 응답 파싱
        def parse_ai_response(response_text):
            try:
                # Extract JSON part from the response
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Parse JSON
                    parsed_json = json.loads(json_str)
                    decision = parsed_json.get('decision')
                    percentage = parsed_json.get('percentage')
                    reason = parsed_json.get('reason')
                    return {'decision': decision, 'percentage': percentage, 'reason': reason}
                else:
                    logger.error("No JSON found in AI response.")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None

        parsed_response = parse_ai_response(response_text)
        if not parsed_response:
            logger.error("Failed to parse AI response.")
            return

        decision = parsed_response.get('decision')
        percentage = parsed_response.get('percentage')
        reason = parsed_response.get('reason')

        if not decision or reason is None:
            logger.error("Incomplete data in AI response.")
            return

        logger.info(f"AI Decision: {decision.upper()}")
        logger.info(f"Percentage: {percentage}")
        logger.info(f"Decision Reason: {reason}")

        order_executed = False

        if decision.lower() == "buy":
            my_usd = usd_balance
            if my_usd is None:
                logger.error("Failed to retrieve USD balance.")
                return
            buy_amount = my_usd * (percentage / 100) * 0.9995  # 수수료 고려
            if buy_amount > 5:  # Bybit은 최소 주문 단위가 다를 수 있음 (예: 5 USD)
                logger.info(f"Buy Order Executed: {percentage}% of available USD")
                try:
                    order = session.place_active_order(
                        symbol="BTCUSD",
                        side="Buy",
                        order_type="Market",
                        qty=buy_amount,
                        time_in_force="GoodTillCancel"
                    )
                    if order and 'result' in order and order['result']:
                        logger.info(f"Buy order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error("Buy order failed.")
                except Exception as e:
                    logger.error(f"Error executing buy order: {e}")
            else:
                logger.warning("Buy Order Failed: Insufficient USD (less than 5 USD)")
        elif decision.lower() == "sell":
            my_btc = btc_balance
            if my_btc is None:
                logger.error("Failed to retrieve BTC balance.")
                return
            sell_amount = my_btc * (percentage / 100)
            try:
                current_price = float(session.latest_information_for_symbol(symbol="BTCUSD")['result'][0]['last_price'])
                if sell_amount * current_price > 5:  # 최소 주문 단위 고려
                    logger.info(f"Sell Order Executed: {percentage}% of held BTC")
                    try:
                        order = session.place_active_order(
                            symbol="BTCUSD",
                            side="Sell",
                            order_type="Market",
                            qty=sell_amount,
                            time_in_force="GoodTillCancel"
                        )
                        if order and 'result' in order and order['result']:
                            order_executed = True
                            logger.info(f"Sell order executed successfully: {order}")
                        else:
                            logger.error("Sell order failed.")
                    except Exception as e:
                        logger.error(f"Error executing sell order: {e}")
                else:
                    logger.warning("Sell Order Failed: Insufficient BTC (less than 5 USD worth)")
            except Exception as e:
                logger.error(f"Error fetching current price: {e}")
        elif decision.lower() == "hold":
            logger.info("Decision is to hold. No action taken.")
        else:
            logger.error("Invalid decision received from AI.")
            return

        # 거래 실행 여부와 관계없이 현재 잔고 조회
        time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
        try:
            wallet = session.get_wallet_balance()
            balances = wallet['result']
            btc_balance = float(next((item['coin_balance'] for item in balances if item['coin'] == 'BTC'), 0))
            usd_balance = float(next((item['coin_balance'] for item in balances if item['coin'] == 'USD'), 0))
            btc_avg_buy_price = float(next((item['avg_price'] for item in balances if item['coin'] == 'BTC'), 0))
            current_btc_price = float(session.latest_information_for_symbol(symbol="BTCUSD")['result'][0]['last_price'])
        except Exception as e:
            logger.error(f"Error fetching updated balances or price: {e}")
            btc_balance = 0
            usd_balance = 0
            btc_avg_buy_price = 0
            current_btc_price = 0

        # 거래 기록을 DB에 저장하기
        trade_data = {
            "timestamp": datetime.utcnow(),
            "decision": decision.lower(),
            "percentage": percentage if order_executed else 0,
            "reason": reason,
            "btc_balance": btc_balance,
            "usd_balance": usd_balance,
            "btc_avg_buy_price": btc_avg_buy_price,
            "btc_usd_price": current_btc_price,
            "reflection": reflection
        }
        log_trade(trades_collection, trade_data)
    except Exception as e:
        logger.error(f"An unexpected error occurred during trading: {e}")
        return

if __name__ == "__main__":
    # 데이터베이스 초기화
    trades_collection = init_db()

    # 중복 실행 방지를 위한 변수
    trading_in_progress = False

    # 트레이딩 작업을 수행하는 함수
    def job():
        global trading_in_progress
        if trading_in_progress:
            logger.warning("Trading job is already in progress, skipping this run.")
            return
        try:
            trading_in_progress = True
            ai_trading(trades_collection)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            trading_in_progress = False

    #테스트
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
