import os
import logging
import time
import json
import re
import schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit import HTTP
import openai
import ta
from ta.utils import dropna
from pymongo import MongoClient  # MongoDB 클라이언트 추가

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bybit 객체 생성
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
if not api_key or not api_secret:
    logger.error("API keys not found. Please check your environment variables.")
    raise ValueError("Missing API keys. Please check your environment variables.")

# Bybit REST API 세션 생성
session = HTTP(
    endpoint="https://api.bybit.com",  # 실제 거래를 위한 엔드포인트
    api_key=api_key,
    api_secret=api_secret
)

# MongoDB 연결 설정
def init_db():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri)
    db = client['bitcoin_trades_db']
    trades_collection = db['trades']
    return trades_collection

# 거래 기록을 DB에 저장하는 함수
def log_trade(trades_collection, decision, percentage, reason, btc_balance,
              usdt_balance, btc_avg_buy_price, btc_usdt_price, reflection=''):
    trade = {
        "timestamp": datetime.now(),
        "decision": decision,
        "percentage": percentage,
        "reason": reason,
        "btc_balance": btc_balance,
        "usdt_balance": usdt_balance,
        "btc_avg_buy_price": btc_avg_buy_price,
        "btc_usdt_price": btc_usdt_price,
        "reflection": reflection
    }
    trades_collection.insert_one(trade)

# 최근 투자 기록 조회
def get_recent_trades(trades_collection, days=7):
    seven_days_ago = datetime.now() - timedelta(days=days)
    cursor = trades_collection.find({"timestamp": {"$gte": seven_days_ago}}).sort(
        "timestamp", -1)
    trades = list(cursor)
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    return trades_df

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (USDT + BTC * 당시 가격)
    initial_balance = trades_df.iloc[-1]['usdt_balance'] + trades_df.iloc[-1][
        'btc_balance'] * trades_df.iloc[-1]['btc_usdt_price']
    # 최종 잔고 계산
    final_balance = trades_df.iloc[0]['usdt_balance'] + trades_df.iloc[0][
        'btc_balance'] * trades_df.iloc[0]['btc_usdt_price']
    return (final_balance - initial_balance) / initial_balance * 100

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are an AI trading assistant tasked with analyzing "
                           "recent trading performance and current market conditions "
                           "to generate insights and improvements for future trading decisions."
            },
            {
                "role": "user",
                "content": f"""
Recent trading data:
{trades_df.to_json(orient='records', date_format='iso')}

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

    try:
        response_content = response.choices[0].message.content
        return response_content
    except (IndexError, AttributeError) as e:
        logger.error(f"Error extracting response content: {e}")
        return None

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df):
    # 볼린저 밴드 추가
    indicator_bb = ta.volatility.BollingerBands(
        close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # RSI (Relative Strength Index) 추가
    df['rsi'] = ta.momentum.RSIIndicator(
        close=df['close'], window=14).rsi()

    # MACD (Moving Average Convergence Divergence) 추가
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # 이동평균선 (단기, 장기)
    df['sma_20'] = ta.trend.SMAIndicator(
        close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(
        close=df['close'], window=12).ema_indicator()

    # Stochastic Oscillator 추가
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Average True Range (ATR) 추가
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    ).average_true_range()

    # On-Balance Volume (OBV) 추가
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['close'],
        volume=df['volume']
    ).on_balance_volume()

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

# 뉴스 데이터 가져오기
def get_bitcoin_news():
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        logger.error("SERPAPI API key is missing.")
        return []  # 빈 리스트 반환
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": "bitcoin OR btc",
        "api_key": serpapi_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        news_results = data.get("news_results", [])
        headlines = []
        for item in news_results:
            headlines.append({
                "title": item.get("title", ""),
                "date": item.get("date", "")
            })

        return headlines[:5]
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []

# 가격 데이터 가져오기 함수 (Bybit용)
def get_ohlcv(symbol, interval, limit):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["retCode"] != 0:
            logger.error(f"Error fetching OHLCV data: {data['retMsg']}")
            return None
        records = data["result"]["list"]
        df = pd.DataFrame(records, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df
    except requests.RequestException as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        return None

### 메인 AI 트레이딩 로직
def ai_trading():
    global session
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    try:
        balances = session.get_wallet_balance(coin="")['result']['balances']
        balances = {balance['coin']: balance for balance in balances}
        filtered_balances = {
            coin: balances.get(coin, {'free': 0, 'availableBalance': 0})
            for coin in ['BTC', 'USDT']
        }
    except Exception as e:
        logger.error(f"Error fetching balances: {e}")
        return

    # 2. 오더북(호가 데이터) 조회
    try:
        orderbook_response = session.orderbook(symbol="BTCUSDT")
        orderbook = orderbook_response['result']
    except Exception as e:
        logger.error(f"Error fetching orderbook: {e}")
        orderbook = None

    # 3. 차트 데이터 조회 및 보조지표 추가
    df_daily = get_ohlcv("BTCUSDT", interval="D", limit=180)
    if df_daily is None:
        logger.error("Failed to retrieve daily OHLCV data.")
        return
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)

    df_hourly = get_ohlcv("BTCUSDT", interval="60", limit=168)  # 7 days of hourly data
    if df_hourly is None:
        logger.error("Failed to retrieve hourly OHLCV data.")
        return
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)

    # 최근 데이터만 사용하도록 설정 (메모리 절약)
    df_daily_recent = df_daily.tail(60)
    df_hourly_recent = df_hourly.tail(48)

    # 4. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()

    # 5. 뉴스 헤드라인 가져오기
    news_headlines = get_bitcoin_news()

    ### AI에게 데이터 제공하고 판단 받기
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None
    try:
        # 데이터베이스 연결
        trades_collection = init_db()

        # 최근 거래 내역 가져오기
        recent_trades = get_recent_trades(trades_collection)

        # 현재 시장 데이터 수집 (기존 코드에서 가져온 데이터 사용)
        current_market_data = {
            "fear_greed_index": fear_greed_index,
            "news_headlines": news_headlines,
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.to_dict(),
            "hourly_ohlcv": df_hourly_recent.to_dict()
        }

        # 반성 및 개선 내용 생성
        reflection = generate_reflection(recent_trades, current_market_data)

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
            model="gpt-3.5-turbo",
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
                    "content": f"""Current investment status: {json.dumps(filtered_balances)}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json()}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json()}
Recent news headlines: {json.dumps(news_headlines)}
Fear and Greed Index: {json.dumps(fear_greed_index)}
"""
                }
            ]
        )

        response_text = response.choices[0].message.content

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
                    return {'decision': decision, 'percentage': percentage,
                            'reason': reason}
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

        # 현재 가격 가져오기
        current_price_data = session.latest_information_for_symbol(symbol="BTCUSDT")
        current_price = float(current_price_data['result']['list'][0]['lastPrice'])

        if decision == "buy":
            my_usdt = float(filtered_balances['USDT']['availableBalance'])
            if my_usdt is None:
                logger.error("Failed to retrieve USDT balance.")
                return
            buy_amount_usdt = my_usdt * (int(percentage) / 100) * 0.9995  # 수수료 고려
            if buy_amount_usdt > 10:  # 최소 거래 금액은 거래소에 따라 다를 수 있음
                logger.info(f"Buy Order Executed: {percentage}% of available USDT")
                try:
                    order_qty = round(buy_amount_usdt / current_price, 6)  # 소수점 자리수는 거래소에 따라 조정
                    order = session.place_order(
                        category="spot",
                        symbol="BTCUSDT",
                        side="Buy",
                        orderType="Market",
                        qty=str(order_qty),
                        timeInForce="GTC"
                    )
                    if order['retCode'] == 0:
                        logger.info(f"Buy order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error(f"Buy order failed: {order['retMsg']}")
                except Exception as e:
                    logger.error(f"Error executing buy order: {e}")
            else:
                logger.warning("Buy Order Failed: Insufficient USDT (less than minimum required)")
        elif decision == "sell":
            my_btc = float(filtered_balances['BTC']['availableBalance'])
            if my_btc is None:
                logger.error("Failed to retrieve BTC balance.")
                return
            sell_amount_btc = my_btc * (int(percentage) / 100) * 0.9995  # 수수료 고려
            if sell_amount_btc * current_price > 10:  # 최소 거래 금액은 거래소에 따라 다를 수 있음
                logger.info(f"Sell Order Executed: {percentage}% of held BTC")
                try:
                    order_qty = round(sell_amount_btc, 6)  # 소수점 자리수는 거래소에 따라 조정
                    order = session.place_order(
                        category="spot",
                        symbol="BTCUSDT",
                        side="Sell",
                        orderType="Market",
                        qty=str(order_qty),
                        timeInForce="GTC"
                    )
                    if order['retCode'] == 0:
                        logger.info(f"Sell order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error(f"Sell order failed: {order['retMsg']}")
                except Exception as e:
                    logger.error(f"Error executing sell order: {e}")
            else:
                logger.warning("Sell Order Failed: Insufficient BTC (less than minimum required)")
        elif decision == "hold":
            logger.info("Decision is to hold. No action taken.")
        else:
            logger.error("Invalid decision received from AI.")
            return

        # 거래 실행 여부와 관계없이 현재 잔고 조회
        time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
        try:
            balances = session.get_wallet_balance(coin="")['result']['balances']
            balances = {balance['coin']: balance for balance in balances}
            btc_balance = float(balances.get('BTC', {'availableBalance': 0})['availableBalance'])
            usdt_balance = float(balances.get('USDT', {'availableBalance': 0})['availableBalance'])
            btc_avg_buy_price = None  # Bybit에서는 평균 매수 단가를 직접 계산해야 함
            current_btc_price = current_price

            # 거래 기록을 DB에 저장하기
            log_trade(
                trades_collection,
                decision,
                int(percentage) if order_executed else 0,
                reason,
                btc_balance,
                usdt_balance,
                btc_avg_buy_price,
                current_btc_price,
                reflection
            )
        except Exception as e:
            logger.error(f"Error fetching updated balances: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return

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
