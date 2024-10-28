import os
from dotenv import load_dotenv
from pybit import HTTP  # 올바른 임포트 경로
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

# .env 파일에 저장된 환경 변수를 불러오기 (API 키 등)
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB 클라이언트 설정
mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    logger.error("MongoDB URI not found. Please check your .env file.")
    raise ValueError("Missing MongoDB URI. Please check your .env file.")
client = MongoClient(mongo_uri)
db = client['bitcoin_trades_db']  # 데이터베이스 이름
trades_collection = db['trades']  # 컬렉션 이름

# Bybit 객체 생성
bybit_api_key = os.getenv("BYBIT_API_KEY")
bybit_api_secret = os.getenv("BYBIT_API_SECRET")
if not bybit_api_key or not bybit_api_secret:
    logger.error("Bybit API keys not found. Please check your .env file.")
    raise ValueError("Missing Bybit API keys. Please check your .env file.")
# Initialize the Bybit HTTP client
bybit = HTTP(
    endpoint="https://api.bybit.com",
    api_key=bybit_api_key,
    api_secret=bybit_api_secret
)

# 거래 기록을 DB에 저장하는 함수 (MongoDB 사용)
def log_trade(decision, percentage, reason, btc_balance, usdt_balance, btc_avg_buy_price, btc_usdt_price, reflection=''):
    trade_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision": decision,
        "percentage": percentage,
        "reason": reason,
        "btc_balance": btc_balance,
        "usdt_balance": usdt_balance,
        "btc_avg_buy_price": btc_avg_buy_price,
        "btc_usdt_price": btc_usdt_price,
        "reflection": reflection
    }
    try:
        trades_collection.insert_one(trade_record)
        logger.info("Trade logged successfully.")
    except Exception as e:
        logger.error(f"Error logging trade to MongoDB: {e}")

# 최근 투자 기록 조회 (MongoDB 사용)
def get_recent_trades(days=7):
    seven_days_ago = (datetime.utcnow() - timedelta(days=days)).isoformat()
    try:
        cursor = trades_collection.find({"timestamp": {"$gt": seven_days_ago}}).sort("timestamp", -1)
        trades = list(cursor)
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame(trades)
    except Exception as e:
        logger.error(f"Error fetching recent trades from MongoDB: {e}")
        return pd.DataFrame()

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    try:
        # 초기 잔고 계산 (USDT + BTC * 현재 가격)
        initial_trade = trades_df.iloc[-1]
        initial_balance = initial_trade['usdt_balance'] + initial_trade['btc_balance'] * initial_trade['btc_usdt_price']
        # 최종 잔고 계산
        final_trade = trades_df.iloc[0]
        final_balance = final_trade['usdt_balance'] + final_trade['btc_balance'] * final_trade['btc_usdt_price']
        return (final_balance - initial_balance) / initial_balance * 100
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        return 0

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None

    openai.api_key = openai_api_key

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 모델 이름
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
{current_market_data}

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
        logger.error(f"Error generating reflection with OpenAI: {e}")
        return None

# --------------------보조지표--------------------

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df):
    try:
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
        
        # Drop any NaN values generated by indicators
        df = dropna(df)
        
        return df
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return df

### 메인 AI 트레이딩 로직
def ai_trading():
    global bybit
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    try:
        # Bybit의 잔고 조회 (USDT 기준)
        balance_info = bybit.get_wallet_balance()
        if not balance_info or 'result' not in balance_info:
            logger.error("Failed to retrieve wallet balance.")
            return
        
        balances = balance_info['result']
        usdt_balance = 0  # Bybit은 USDT를 사용
        btc_balance = 0
        btc_avg_buy_price = 0
        for balance in balances:
            if balance['coin'] == 'BTC':
                btc_balance = float(balance['wallet_balance'])
                btc_avg_buy_price = float(balance.get('avgPrice', 0))  # avgPrice 필드는 일부 API에서 제공
            elif balance['coin'] == 'USDT':
                usdt_balance = float(balance['wallet_balance'])
        
        # 2. 오더북(호가 데이터) 조회
        orderbook = bybit.order_book(symbol="BTCUSDT")  # Bybit 심볼 형식 변경
        
        # 3. 차트 데이터 조회 및 보조지표 추가
        # Bybit의 차트 데이터는 REST API를 통해 가져올 수 있습니다.
        # 여기서는 REST API 사용 예시를 제공합니다.
        def get_ohlcv(symbol, interval, limit):
            try:
                response = bybit.query_kline(symbol=symbol, interval=interval, limit=limit)
                if response['ret_code'] != 0:
                    logger.error(f"Error fetching OHLCV data: {response['ret_msg']}")
                    return pd.DataFrame()
                data = response['result']
                df = pd.DataFrame(data)
                df.rename(columns={
                    'open_time': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }, inplace=False)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                return df
            except Exception as e:
                logger.error(f"Exception fetching OHLCV data: {e}")
                return pd.DataFrame()
        
        # 일간 데이터 (1D)
        df_daily = get_ohlcv("BTCUSDT", interval=1, limit=180)
        df_daily = dropna(df_daily)
        df_daily = add_indicators(df_daily)
        
        # 시간별 데이터 (1H)
        df_hourly = get_ohlcv("BTCUSDT", interval=60, limit=168)  # 7 days of hourly data
        df_hourly = dropna(df_hourly)
        df_hourly = add_indicators(df_hourly)
    
        # 최근 데이터만 사용하도록 설정 (메모리 절약)
        df_daily_recent = df_daily.tail(60)
        df_hourly_recent = df_hourly.tail(48)
        
        # 4. 공포 탐욕 지수 가져오기
        fear_greed_index = get_fear_and_greed_index()
        
        # 5. 뉴스 헤드라인 가져오기
        news_headlines = get_bitcoin_news()
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return
    
    ### AI에게 데이터 제공하고 판단 받기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None

    try:
        # 최근 거래 내역 가져오기
        recent_trades = get_recent_trades()
        
        # 현재 시장 데이터 수집 (기존 코드에서 가져온 데이터 사용)
        current_market_data = {
            "fear_greed_index": fear_greed_index,
            "news_headlines": news_headlines,
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.to_dict(orient='records'),
            "hourly_ohlcv": df_hourly_recent.to_dict(orient='records')
        }
        
        # 반성 및 개선 내용 생성
        reflection = generate_reflection(recent_trades, current_market_data)
        
        if not reflection:
            logger.error("Failed to generate reflection.")
            return
        
        # AI 모델에 반성 내용 제공
        # Few-shot prompting으로 JSON 예시 추가
        examples = """
Example Response 1:
{
  "decision": "buy",
  "percentage": 50,
  "reason": "Based on the current market indicators and positive news, it's a good opportunity to invest."
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
            model="gpt-4",  # 모델 이름 수정
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert in Bitcoin investing. This analysis is performed every 4 hours. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:

- Technical indicators and market data
- Recent news headlines and their potential impact on Bitcoin price
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
                    "content": f"""Current investment status: {json.dumps({'BTC': btc_balance, 'USDT': usdt_balance})}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json(orient='records')}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json(orient='records')}
Recent news headlines: {json.dumps(news_headlines)}
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
            my_usdt = usdt_balance  # Bybit은 USDT 사용
            if my_usdt is None:
                logger.error("Failed to retrieve USDT balance.")
                return
            buy_amount_usdt = my_usdt * (percentage / 100) * 0.9995  # 수수료 고려
            if buy_amount_usdt > 5:  # Bybit 최소 주문 금액을 USDT 기준으로 설정 (예: 5 USDT)
                logger.info(f"Buy Order Executed: {percentage}% of available USDT")
                try:
                    # 현재 BTC 가격을 가져와 USDT를 BTC로 변환
                    current_price_info = bybit.latest_information_for_symbol(symbol="BTCUSDT")
                    if not current_price_info or 'result' not in current_price_info:
                        logger.error("Failed to retrieve current BTC price.")
                        return
                    current_price = float(current_price_info['result'][0]['last_price'])
                    buy_qty = buy_amount_usdt / current_price  # USDT를 BTC 수량으로 변환

                    order = bybit.place_active_order(
                        symbol="BTCUSDT",
                        side="Buy",
                        order_type="Market",
                        qty=round(buy_qty, 6),  # Bybit은 소수점 6자리까지 지원
                        time_in_force="GoodTillCancel"
                    )
                    if order and order.get('ret_code') == 0:
                        logger.info(f"Buy order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error(f"Buy order failed: {order.get('ret_msg')}")
                except Exception as e:
                    logger.error(f"Error executing buy order: {e}")
            else:
                logger.warning("Buy Order Failed: Insufficient USDT (less than 5 USDT)")
        elif decision.lower() == "sell":
            my_btc = btc_balance
            if my_btc is None:
                logger.error("Failed to retrieve BTC balance.")
                return
            sell_amount_btc = my_btc * (percentage / 100)
            # 현재 BTC 가격을 가져와 USDT 금액을 확인
            current_price_info = bybit.latest_information_for_symbol(symbol="BTCUSDT")
            if not current_price_info or 'result' not in current_price_info:
                logger.error("Failed to retrieve current BTC price.")
                return
            current_price = float(current_price_info['result'][0]['last_price'])
            if sell_amount_btc * current_price > 5:  # 최소 주문 금액
                logger.info(f"Sell Order Executed: {percentage}% of held BTC")
                try:
                    order = bybit.place_active_order(
                        symbol="BTCUSDT",
                        side="Sell",
                        order_type="Market",
                        qty=round(sell_amount_btc, 6),  # Bybit은 소수점 6자리까지 지원
                        time_in_force="GoodTillCancel"
                    )
                    if order and order.get('ret_code') == 0:
                        logger.info(f"Sell order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error(f"Sell order failed: {order.get('ret_msg')}")
                except Exception as e:
                    logger.error(f"Error executing sell order: {e}")
            else:
                logger.warning("Sell Order Failed: Insufficient BTC (less than 5 USDT worth)")
        elif decision.lower() == "hold":
            logger.info("Decision is to hold. No action taken.")
        else:
            logger.error("Invalid decision received from AI.")
            return

        # 거래 실행 여부와 관계없이 현재 잔고 조회
        time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
        try:
            # Update balances after trade
            balance_info = bybit.get_wallet_balance()
            if balance_info and 'result' in balance_info:
                balances = balance_info['result']
                btc_balance = 0
                usdt_balance = 0
                btc_avg_buy_price = 0
                for balance in balances:
                    if balance['coin'] == 'BTC':
                        btc_balance = float(balance['wallet_balance'])
                        btc_avg_buy_price = float(balance.get('avgPrice', 0))
                    elif balance['coin'] == 'USDT':
                        usdt_balance = float(balance['wallet_balance'])
            current_btc_price_info = bybit.latest_information_for_symbol(symbol="BTCUSDT")
            if current_btc_price_info and 'result' in current_btc_price_info:
                current_btc_price = float(current_btc_price_info['result'][0]['last_price'])
            else:
                current_btc_price = 0
        except Exception as e:
            logger.error(f"Error fetching updated balances: {e}")
            return

        # 거래 기록을 DB에 저장하기
        log_trade(
            decision,
            percentage if order_executed else 0,
            reason,
            btc_balance,
            usdt_balance,
            btc_avg_buy_price,
            current_btc_price,
            reflection
        )
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return

# 보조 함수들

def get_fear_and_greed_index():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        data = response.json()
        return data['data'][0]['value']
    except Exception as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")
        return None

def get_bitcoin_news():
    try:
        # 예시로 News API 사용 (API 키 필요)
        news_api_key = os.getenv("NEWS_API_KEY")
        if not news_api_key:
            logger.error("News API key not found.")
            return []
        url = f"https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&apiKey={news_api_key}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        headlines = [article['title'] for article in articles]
        return headlines
    except Exception as e:
        logger.error(f"Error fetching Bitcoin news: {e}")
        return []

if __name__ == "__main__":
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
            ai_trading()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            trading_in_progress = False
    
    #테스트
    # job()
    
    # 매 4시간마다 실행 (Bybit는 UTC 시간 기준으로 설정 필요할 수 있음)
    schedule.every(4).hours.do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
