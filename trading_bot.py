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
import schedule
import threading

# 환경 변수 로드
load_dotenv()

# 로깅 설정 (콘솔과 파일 모두 로그를 출력하도록 설정)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log")  # 로그를 파일로 저장
    ]
)
logger = logging.getLogger(__name__)

# 글로벌 변수 설정
trades_collection = None
bybit = None
trading_in_progress = False

# MongoDB 설정 및 연결
def init_db():
    global trades_collection
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        logger.critical("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
        raise ValueError("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
    try:
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        logger.info("MongoDB 연결 완료!")
    except Exception as e:
        logger.critical(f"MongoDB 연결 오류: {e}")
        raise

# Bybit API 설정
def setup_bybit():
    global bybit
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
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

# ... [Rest of your existing functions: get_account_balance, log_balance_to_mongodb, etc.] ...

# AI 트레이딩 로직
def ai_trading():
    global trades_collection, bybit
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    try:
        balance_data = get_account_balance()
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
                
                # 최근 데이터만 사용하도록 설정 (메모리 절약) 및 결측치 제거
                df_daily_recent = df_daily.tail(60).dropna()
                df_hourly_recent = df_hourly.tail(48).dropna()
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
    
    ### AI에게 데이터 제공하고 판단 받기
    try:
        # 최근 거래 내역 가져오기
        recent_trades = get_recent_trades()
        
        # 현재 시장 데이터 수집
        current_market_data = {
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.to_dict(orient='records'),
            "hourly_ohlcv": df_hourly_recent.to_dict(orient='records')
        }
        
        # 반성 및 개선 내용 생성
        reflection = generate_reflection(recent_trades, current_market_data)
        
        if not reflection:
            logger.error("Reflection 생성 실패. AI 트레이딩 로직 중단.")
            return
        
        # AI 모델에 반성 내용 제공 및 결정 생성
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
  "reason": "Due to negative trends in the market, it is advisable to reduce holdings."
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
                    "content": f"""You are an expert in Bitcoin investing. This analysis is performed every 4 hours. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:

- Technical indicators and market data
- Overall market sentiment
- Recent trading performance and reflection

Recent trading reflection:
{reflection}

Based on your analysis, make a decision and provide your reasoning.

Please provide your response in the following JSON format:

{examples}

Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
Your percentage should reflect the strength of your conviction in the decision based on the analyzed data."""
                },
                {
                    "role": "user",
                    "content": f"""Current investment status: {json.dumps([balance for balance in bybit.get_wallet_balance().get('result', {}).get('coin', []) if balance['coin'] in ['USDT']])}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json(orient='records')}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json(orient='records')}
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
                # Bybit 선물에서는 실제 BTC 잔고가 없으므로, 포지션을 종료하여 USDT를 회수합니다.
                # 현재 포지션 조회
                position = bybit.get_position(symbol="BTCUSDT")
                if position['ret_code'] == 0 and 'result' in position and len(position['result']) > 0:
                    current_position = position['result'][0]
                    size = float(current_position.get('size', 0))
                    if size > 0:
                        sell_size = size * (percentage / 100)
                        if sell_size > 0:
                            logger.info(f"Sell Order Executed: {percentage}% of current position size")
                            order = bybit.place_active_order(
                                symbol="BTCUSDT",
                                side="Sell",
                                order_type="Market",
                                qty=sell_size,
                                time_in_force="GoodTillCancel"
                            )
                            if order and order['ret_code'] == 0:
                                logger.info(f"Sell order executed successfully: {order}")
                                order_executed = True
                            else:
                                logger.error(f"Sell order failed: {order.get('ret_msg', 'No ret_msg')}")
                        else:
                            logger.warning("Sell Order Failed: Calculated sell size is zero.")
                    else:
                        logger.info("No open position to sell.")
                else:
                    logger.error(f"포지션 조회 실패 또는 포지션 없음: {position.get('ret_msg', 'No ret_msg')}")
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
            balance_data = get_account_balance()
            if balance_data:
                usdt_balance = balance_data['available_to_withdraw']
                # 현재 BTC 가격 조회
                current_btc_price = get_current_price_bybit("BTCUSDT")

                # 거래 기록을 DB에 저장하기
                log_trade(
                    decision, 
                    percentage if order_executed else 0, 
                    reason, 
                    usdt_balance, 
                    current_btc_price, 
                    reflection
                )
        except Exception as e:
            logger.error(f"잔고 조회 및 거래 기록 저장 오류: {e}")

# Scheduler setup
def run_scheduler():
    schedule.every().hour.do(ai_trading)
    logger.info("Scheduler started: ai_trading will run every hour.")
    while True:
        schedule.run_pending()
        time.sleep(1)

# Main function
def main():
    # Initialize MongoDB and Bybit connections
    init_db()
    setup_bybit()
    
    # Run ai_trading once at startup
    logger.info("Running initial ai_trading execution.")
    ai_trading()
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user.")

if __name__ == "__main__":
    main()
