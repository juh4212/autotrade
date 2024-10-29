import os
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
from pybit import HTTP  # Bybit v5 API
from dotenv import load_dotenv
import pandas as pd
import openai
import re
import requests
import ta
import time
import json

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# MongoDB 설정 및 연결
def setup_mongodb():
    mongo_uri = os.getenv("MONGODB_URI")
    print("MongoDB URI:", mongo_uri)  # URI 확인을 위해 출력
    try:
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        logger.info("MongoDB 연결 완료!")
        return trades_collection
    except Exception as e:
        logger.critical(f"MongoDB 연결 오류: {e}")
        raise

# Bybit API 설정
def setup_bybit():
    try:
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.critical("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
            raise ValueError("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
        bybit = HTTP(
            endpoint="https://api.bybit.com",
            api_key=api_key,
            api_secret=api_secret
        )
        logger.info("Bybit API 연결 완료!")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

# Bybit 계좌 잔고 조회
def get_account_balance(bybit):
    try:
        # Wallet Balance 조회 (통합 계좌)
        wallet_balance = bybit.get_wallet_balance(coin=None, accountType="UNIFIED")
        logger.info("Bybit API 응답 데이터: %s", wallet_balance)  # 전체 응답 데이터 출력

        if wallet_balance['ret_code'] == 0 and 'result' in wallet_balance:
            account_info = wallet_balance['result']
            # USDT 잔고 정보 추출
            usdt_balance = account_info.get('USDT', {})
            equity = float(usdt_balance.get('equity', 0))
            available_to_withdraw = float(usdt_balance.get('available_balance', 0))

            logger.info(f"USDT 전체 자산: {equity}, 사용 가능한 자산: {available_to_withdraw}")
            return {
                "equity": equity,
                "available_to_withdraw": available_to_withdraw
            }
        else:
            logger.error(f"잔고 데이터를 가져오지 못했습니다: {wallet_balance.get('ret_msg', 'No ret_msg')}")
            return None
    except Exception as e:
        logger.error(f"Bybit 잔고 조회 오류: {e}")
        return None

# MongoDB에 잔고 기록
def log_balance_to_mongodb(collection, balance_data):
    balance_record = {
        "timestamp": datetime.utcnow(),
        "equity": balance_data["equity"],
        "available_to_withdraw": balance_data["available_to_withdraw"]
    }
    try:
        collection.insert_one(balance_record)
        logger.info("계좌 잔고가 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"MongoDB에 계좌 잔고 저장 오류: {e}")

# 거래 기록을 DB에 저장하는 함수
def log_trade(collection, decision, percentage, reason, btc_balance, usdt_balance, btc_avg_buy_price, btc_usdt_price, reflection=''):
    trade_record = {
        "timestamp": datetime.utcnow(),
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
        collection.insert_one(trade_record)
        logger.info("거래 기록이 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"MongoDB에 거래 기록 저장 오류: {e}")

# 최근 투자 기록 조회
def get_recent_trades(collection, days=7):
    seven_days_ago = datetime.utcnow() - timedelta(days=days)
    try:
        cursor = collection.find({"timestamp": {"$gt": seven_days_ago}}).sort("timestamp", -1)
        trades = list(cursor)
        if not trades:
            logger.info("최근 거래 기록이 없습니다.")
            return pd.DataFrame()
        # Convert MongoDB documents to DataFrame
        trades_df = pd.DataFrame(trades)
        return trades_df
    except Exception as e:
        logger.error(f"최근 거래 기록 조회 오류: {e}")
        return pd.DataFrame()

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (USDT + BTC * 현재 가격)
    initial_trade = trades_df.iloc[-1]
    initial_balance = initial_trade['usdt_balance'] + initial_trade['btc_balance'] * initial_trade['btc_usdt_price']
    # 최종 잔고 계산
    final_trade = trades_df.iloc[0]
    final_balance = final_trade['usdt_balance'] + final_trade['btc_balance'] * final_trade['btc_usdt_price']
    return (final_balance - initial_balance) / initial_balance * 100

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key가 설정되지 않았거나 유효하지 않습니다.")
        return None

    openai.api_key = openai_api_key

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."},
                {"role": "user", "content": f"""
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
"""}
            ],
            max_tokens=500
        )
        reflection = response['choices'][0]['message']['content'].strip()
        return reflection
    except Exception as e:
        logger.error(f"OpenAI reflection 생성 오류: {e}")
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

# 뉴스 데이터 가져오기
def get_bitcoin_news():
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        logger.error("SERPAPI API key가 설정되지 않았습니다.")
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

# 현재 가격 조회 함수 (Bybit)
def get_current_price_bybit(bybit, symbol):
    try:
        ticker = bybit.query_public("v5/market/ticker", {"symbol": symbol})
        if ticker['ret_code'] == 0:
            return float(ticker['result']['list'][0]['last_price'])
        else:
            logger.error(f"가격 조회 실패: {ticker.get('ret_msg', 'No ret_msg')}")
            return 0
    except Exception as e:
        logger.error(f"가격 조회 오류: {e}")
        return 0

# AI 트레이딩 로직
def ai_trading(trades_collection, bybit):
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    try:
        balance_data = get_account_balance(bybit)
        if not balance_data:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return
    except Exception as e:
        logger.error(f"잔고 조회 오류: {e}")
        return

    # 2. 오더북(호가 데이터) 조회
    try:
        # Bybit API v5에서 오더북 조회
        orderbook_response = bybit.orderbook(symbol="BTCUSDT")
        if orderbook_response['ret_code'] == 0:
            orderbook = orderbook_response['result']
        else:
            logger.error(f"오더북 조회 실패: {orderbook_response.get('ret_msg', 'No ret_msg')}")
            orderbook = {}
    except Exception as e:
        logger.error(f"오더북 조회 오류: {e}")
        orderbook = {}

    # 3. 차트 데이터 조회 및 보조지표 추가
    try:
        # Bybit API v5에서 Kline 데이터 조회
        klines_response_daily = bybit.query_public("v5/market/kline", {
            "symbol": "BTCUSDT",
            "interval": "D",
            "limit": 180
        })
        if klines_response_daily['ret_code'] == 0:
            klines_daily = klines_response_daily['result']['list']
            df_daily = pd.DataFrame(klines_daily)
            df_daily.rename(columns={
                'open_time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'turnover': 'turnover'
            }, inplace=True)
            df_daily['close'] = df_daily['close'].astype(float)
            df_daily['high'] = df_daily['high'].astype(float)
            df_daily['low'] = df_daily['low'].astype(float)
            df_daily['volume'] = df_daily['volume'].astype(float)
            df_daily = add_indicators(df_daily)
            
            klines_response_hourly = bybit.query_public("v5/market/kline", {
                "symbol": "BTCUSDT",
                "interval": "60",
                "limit": 168
            })
            if klines_response_hourly['ret_code'] == 0:
                klines_hourly = klines_response_hourly['result']['list']
                df_hourly = pd.DataFrame(klines_hourly)
                df_hourly.rename(columns={
                    'open_time': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'turnover': 'turnover'
                }, inplace=True)
                df_hourly['close'] = df_hourly['close'].astype(float)
                df_hourly['high'] = df_hourly['high'].astype(float)
                df_hourly['low'] = df_hourly['low'].astype(float)
                df_hourly['volume'] = df_hourly['volume'].astype(float)
                df_hourly = add_indicators(df_hourly)
                
                # 최근 데이터만 사용하도록 설정 (메모리 절약)
                df_daily_recent = df_daily.tail(60)
                df_hourly_recent = df_hourly.tail(48)
            else:
                logger.error(f"시간별 클라인 조회 실패: {klines_response_hourly.get('ret_msg', 'No ret_msg')}")
                df_hourly_recent = pd.DataFrame()
        else:
            logger.error(f"일별 클라인 조회 실패: {klines_response_daily.get('ret_msg', 'No ret_msg')}")
            df_daily_recent = pd.DataFrame()
            df_hourly_recent = pd.DataFrame()
    except Exception as e:
        logger.error(f"차트 데이터 조회 오류: {e}")
        df_daily_recent = pd.DataFrame()
        df_hourly_recent = pd.DataFrame()
    
    # 4. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()
    
    # 5. 뉴스 헤드라인 가져오기
    news_headlines = get_bitcoin_news()
    
    ### AI에게 데이터 제공하고 판단 받기
    try:
        # 최근 거래 내역 가져오기
        recent_trades = get_recent_trades(trades_collection)
        
        # 현재 시장 데이터 수집
        current_market_data = {
            "fear_greed_index": fear_greed_index,
            "news_headlines": news_headlines,
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.to_dict(orient='records'),
            "hourly_ohlcv": df_hourly_recent.to_dict(orient='records')
        }
        
        # 반성 및 개선 내용 생성
        reflection = generate_reflection(recent_trades, current_market_data)
        
        # AI 모델에 반성 내용 제공 및 결정 생성
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
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Bitcoin investing. This analysis is performed every 4 hours. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:\n\n- Technical indicators and market data\n- Recent news headlines and their potential impact on Bitcoin price\n- The Fear and Greed Index and its implications\n- Overall market sentiment\n- Recent trading performance and reflection\n\nRecent trading reflection:\n{reflection}\n\nBased on your analysis, make a decision and provide your reasoning.\n\nPlease provide your response in the following JSON format:\n\n{examples}\n\nEnsure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.\nYour percentage should reflect the strength of your conviction in the decision based on the analyzed data."
                },
                {
                    "role": "user",
                    "content": f"""Current investment status: {json.dumps([balance for balance in bybit.get_wallet_balance().get('result', {}).get('coin', []) if balance['coin'] in ['BTC', 'USDT']])}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json(orient='records')}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json(orient='records')}
Recent news headlines: {json.dumps(news_headlines)}
Fear and Greed Index: {json.dumps(fear_greed_index)}
"""
                }
            ],
            max_tokens=500
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
            try:
                buy_amount = balance_data['available_to_withdraw'] * (percentage / 100) * 0.9995  # 수수료 고려
                if buy_amount > 5:  # Bybit은 최소 주문 금액을 USDT 기준으로 설정
                    logger.info(f"Buy Order Executed: {percentage}% of available USDT")
                    order = bybit.place_active_order(
                        symbol="BTCUSDT",
                        side="Buy",
                        order_type="Market",
                        qty=buy_amount,
                        time_in_force="GoodTillCancel"
                    )
                    if order and order['ret_code'] == 0:
                        logger.info(f"Buy order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error(f"Buy order failed: {order.get('ret_msg', 'No ret_msg')}")
                else:
                    logger.warning("Buy Order Failed: Insufficient USDT (less than 5 USDT)")
            except Exception as e:
                logger.error(f"Error executing buy order: {e}")
        elif decision.lower() == "sell":
            try:
                # Bybit에서는 USDT 잔고가 없으므로, BTC 잔고를 사용
                btc_balance = balance_data.get('equity', 0)  # Bybit의 'equity'가 BTC 잔고에 해당
                sell_amount = btc_balance * (percentage / 100)
                current_price = get_current_price_bybit(bybit, "BTCUSDT")
                if sell_amount * current_price > 5:  # 최소 주문 금액
                    logger.info(f"Sell Order Executed: {percentage}% of held BTC")
                    order = bybit.place_active_order(
                        symbol="BTCUSDT",
                        side="Sell",
                        order_type="Market",
                        qty=sell_amount,
                        time_in_force="GoodTillCancel"
                    )
                    if order and order['ret_code'] == 0:
                        logger.info(f"Sell order executed successfully: {order}")
                        order_executed = True
                    else:
                        logger.error(f"Sell order failed: {order.get('ret_msg', 'No ret_msg')}")
                else:
                    logger.warning("Sell Order Failed: Insufficient BTC (less than 5 USDT worth)")
            except Exception as e:
                logger.error(f"Error executing sell order: {e}")
        elif decision.lower() == "hold":
            logger.info("Decision is to hold. No action taken.")
        else:
            logger.error("Invalid decision received from AI.")
            return
    
        # 거래 실행 여부와 관계없이 현재 잔고 조회
        try:
            time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
            balance_data = get_account_balance(bybit)
            if balance_data:
                usdt_balance = balance_data['available_to_withdraw']
                # 현재 BTC 가격 조회
                current_btc_price = get_current_price_bybit(bybit, "BTCUSDT")
    
                # 거래 기록을 DB에 저장하기
                log_trade(
                    trades_collection, 
                    decision, 
                    percentage if order_executed else 0, 
                    reason, 
                    balance_data.get('btc_balance', 0),  # Bybit의 BTC 잔고 필드에 맞게 조정
                    usdt_balance, 
                    balance_data.get('btc_avg_buy_price', 0), 
                    current_btc_price, 
                    reflection
                )
        except Exception as e:
            logger.error(f"잔고 조회 및 거래 기록 저장 오류: {e}")

# 데이터프레임에서 결측치 제거 함수
def dropna(df):
    return df.dropna()

# 메인 실행
if __name__ == "__main__":
    try:
        # MongoDB와 Bybit 연결 설정
        trades_collection = setup_mongodb()
        bybit = setup_bybit()
    
        # 초기 잔고 기록
        balance_data = get_account_balance(bybit)
        if balance_data:
            log_balance_to_mongodb(trades_collection, balance_data)
    
        # AI 트레이딩 실행
        ai_trading(trades_collection, bybit)
    
    except Exception as e:
        logger.critical(f"시스템 오류: {e}")
