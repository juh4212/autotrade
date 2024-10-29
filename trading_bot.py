import os
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
from pybit.unified_trading import HTTP  # Bybit v5 API를 사용 중임을 가정
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
    except Exception as e:
        logger.critical(f"MongoDB 클라이언트 생성 오류: {e}")
        raise
    try:
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
    except Exception as e:
        logger.critical(f"MongoDB 데이터베이스/컬렉션 접근 오류: {e}")
        raise
    else:
        logger.info("MongoDB 연결 완료!")
    finally:
        # 필요 시 클라이언트 종료 코드 추가 가능
        pass

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
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise
    else:
        logger.info("Bybit API 연결 완료!")
    finally:
        # 필요 시 추가 정리 코드
        pass

# Bybit 계좌 잔고 조회
def get_account_balance():
    global bybit
    try:
        wallet_balance = bybit.get_wallet_balance(coin=None, accountType="UNIFIED")
    except requests.exceptions.RequestException as e:
        logger.error(f"네트워크 오류로 인한 잔고 조회 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"잔고 조회 중 예상치 못한 오류 발생: {e}")
        return None
    else:
        logger.info("Bybit API 응답 데이터: %s", wallet_balance)  # 전체 응답 데이터 출력
        if wallet_balance['ret_code'] == 0 and 'result' in wallet_balance:
            account_info = wallet_balance['result']
            usdt_balance = account_info.get('USDT', {})
            equity = float(usdt_balance.get('equity', 0))
            available_to_withdraw = float(usdt_balance.get('available_balance', 0))
            logger.info(f"USDT 전체 자산 (Equity): {equity}, 사용 가능한 자산 (Available to Withdraw): {available_to_withdraw}")
            return {
                "equity": equity,
                "available_to_withdraw": available_to_withdraw
            }
        else:
            logger.error(f"잔고 데이터를 가져오지 못했습니다: {wallet_balance.get('ret_msg', 'No ret_msg')}")
            return None
    finally:
        logger.debug("get_account_balance 함수 종료.")

# MongoDB에 잔고 기록
def log_balance_to_mongodb(balance_data):
    global trades_collection
    balance_record = {
        "timestamp": datetime.utcnow(),
        "equity": balance_data["equity"],
        "available_to_withdraw": balance_data["available_to_withdraw"]
    }
    try:
        trades_collection.insert_one(balance_record)
    except Exception as e:
        logger.error(f"MongoDB에 계좌 잔고 저장 오류: {e}")
    else:
        logger.info("계좌 잔고가 MongoDB에 성공적으로 저장되었습니다.")
    finally:
        logger.debug("log_balance_to_mongodb 함수 종료.")

# 거래 기록을 DB에 저장하는 함수
def log_trade(decision, percentage, reason, usdt_balance, btc_usdt_price, reflection=''):
    global trades_collection
    trade_record = {
        "timestamp": datetime.utcnow(),
        "decision": decision,
        "percentage": percentage,
        "reason": reason,
        "usdt_balance": usdt_balance,
        "btc_usdt_price": btc_usdt_price,
        "reflection": reflection
    }
    try:
        trades_collection.insert_one(trade_record)
    except Exception as e:
        logger.error(f"MongoDB에 거래 기록 저장 오류: {e}")
    else:
        logger.info("거래 기록이 MongoDB에 성공적으로 저장되었습니다.")
    finally:
        logger.debug("log_trade 함수 종료.")

# 최근 투자 기록 조회
def get_recent_trades(days=7):
    global trades_collection
    seven_days_ago = datetime.utcnow() - timedelta(days=days)
    try:
        cursor = trades_collection.find({"timestamp": {"$gt": seven_days_ago}}).sort("timestamp", -1)
        trades = list(cursor)
    except Exception as e:
        logger.error(f"최근 거래 기록 조회 오류: {e}")
        return pd.DataFrame()
    else:
        if not trades:
            logger.info("최근 거래 기록이 없습니다.")
            return pd.DataFrame()
        trades_df = pd.DataFrame(trades)
        return trades_df
    finally:
        logger.debug("get_recent_trades 함수 종료.")

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    initial_trade = trades_df.iloc[-1]
    initial_balance = initial_trade['equity']
    final_trade = trades_df.iloc[0]
    final_balance = final_trade['equity']
    performance = (final_balance - initial_balance) / initial_balance * 100
    logger.info(f"Calculated performance: {performance:.2f}%")
    return performance

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key가 설정되지 않았거나 유효하지 않습니다.")
        return None

    openai.api_key = openai_api_key

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
            ],
            max_tokens=500
        )
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API 오류: {e}")
        return None
    except Exception as e:
        logger.error(f"Reflection 생성 중 예상치 못한 오류 발생: {e}")
        return None
    else:
        try:
            reflection = response['choices'][0]['message']['content'].strip()
            logger.info("Reflection 생성 성공.")
            return reflection
        except (KeyError, IndexError) as e:
            logger.error(f"Reflection 파싱 오류: {e}")
            return None
    finally:
        logger.debug("generate_reflection 함수 종료.")

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
    except Exception as e:
        logger.error(f"보조 지표 추가 중 오류 발생: {e}")
    else:
        logger.info("보조 지표가 성공적으로 추가되었습니다.")
    finally:
        return df

# 현재 가격 조회 함수 (Bybit)
def get_current_price_bybit(symbol):
    global bybit
    try:
        ticker = bybit.query_public("v5/market/ticker", {"symbol": symbol})
    except requests.exceptions.RequestException as e:
        logger.error(f"네트워크 오류로 인한 가격 조회 실패: {e}")
        return 0
    except Exception as e:
        logger.error(f"가격 조회 중 예상치 못한 오류 발생: {e}")
        return 0
    else:
        if ticker['ret_code'] == 0:
            try:
                last_price = float(ticker['result']['list'][0]['last_price'])
                logger.info(f"현재 {symbol} 가격: {last_price}")
                return last_price
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"가격 데이터 파싱 오류: {e}")
                return 0
        else:
            logger.error(f"가격 조회 실패: {ticker.get('ret_msg', 'No ret_msg')}")
            return 0
    finally:
        logger.debug("get_current_price_bybit 함수 종료.")

# AI 트레이딩 로직
def ai_trading():
    global trades_collection, bybit
    logger.info("AI 트레이딩 로직 시작.")
    try:
        ### 데이터 가져오기
        # 1. 현재 투자 상태 조회
        balance_data = get_account_balance()
        if not balance_data:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return

        # 2. 오더북(호가 데이터) 조회
        try:
            orderbook_response = bybit.orderbook(symbol="BTCUSDT")
        except requests.exceptions.RequestException as e:
            logger.error(f"네트워크 오류로 인한 오더북 조회 실패: {e}")
            orderbook = {}
        except Exception as e:
            logger.error(f"오더북 조회 중 예상치 못한 오류 발생: {e}")
            orderbook = {}
        else:
            if orderbook_response['ret_code'] == 0:
                orderbook = orderbook_response['result']
            else:
                logger.error(f"오더북 조회 실패: {orderbook_response.get('ret_msg', 'No ret_msg')}")
                orderbook = {}
        finally:
            logger.debug("오더북 조회 완료.")

        # 3. 차트 데이터 조회 및 보조지표 추가
        try:
            # Bybit API v5에서 Kline 데이터 조회
            klines_response_daily = bybit.query_public("v5/market/kline", {
                "symbol": "BTCUSDT",
                "interval": "D",
                "limit": 180
            })
            klines_response_hourly = bybit.query_public("v5/market/kline", {
                "symbol": "BTCUSDT",
                "interval": "60",
                "limit": 168
            })
        except requests.exceptions.RequestException as e:
            logger.error(f"네트워크 오류로 인한 클라인 데이터 조회 실패: {e}")
            df_daily_recent = pd.DataFrame()
            df_hourly_recent = pd.DataFrame()
        except Exception as e:
            logger.error(f"클라인 데이터 조회 중 예상치 못한 오류 발생: {e}")
            df_daily_recent = pd.DataFrame()
            df_hourly_recent = pd.DataFrame()
        else:
            # 일별 클라인 데이터 처리
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
                df_daily[['close', 'high', 'low', 'volume']] = df_daily[['close', 'high', 'low', 'volume']].astype(float)
                df_daily = add_indicators(df_daily)
            else:
                logger.error(f"일별 클라인 조회 실패: {klines_response_daily.get('ret_msg', 'No ret_msg')}")
                df_daily = pd.DataFrame()

            # 시간별 클라인 데이터 처리
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
                df_hourly[['close', 'high', 'low', 'volume']] = df_hourly[['close', 'high', 'low', 'volume']].astype(float)
                df_hourly = add_indicators(df_hourly)
            else:
                logger.error(f"시간별 클라인 조회 실패: {klines_response_hourly.get('ret_msg', 'No ret_msg')}")
                df_hourly = pd.DataFrame()
        finally:
            # 최근 데이터만 사용하도록 설정 (메모리 절약) 및 결측치 제거
            if 'df_daily' in locals() and not df_daily.empty:
                df_daily_recent = df_daily.tail(60).dropna()
            else:
                df_daily_recent = pd.DataFrame()
            if 'df_hourly' in locals() and not df_hourly.empty:
                df_hourly_recent = df_hourly.tail(48).dropna()
            else:
                df_hourly_recent = pd.DataFrame()
            logger.debug("클라인 데이터 처리 완료.")

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
        except Exception as e:
            logger.error(f"AI 판단 준비 중 오류 발생: {e}")
            return
        else:
            logger.info("AI 판단을 위한 데이터 준비 완료.")
        finally:
            logger.debug("AI 판단 준비 단계 종료.")

        # AI 모델에 반성 내용 제공 및 결정 생성
        try:
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
                        "content": f"""Current investment status: {json.dumps([balance for balance in get_account_balance().get('USDT', {}) if balance.get('coin') == 'USDT'])}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json(orient='records')}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json(orient='records')}
"""
                    }
                ],
                max_tokens=500
            )
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API 오류 during decision making: {e}")
        return
    except Exception as e:
        logger.error(f"결정 생성 중 예상치 못한 오류 발생: {e}")
        return
    else:
        try:
            response_text = response['choices'][0]['message']['content']
            logger.debug(f"AI 응답: {response_text}")
        except (KeyError, IndexError) as e:
            logger.error(f"AI 응답 파싱 오류: {e}")
            return
    finally:
        logger.debug("AI 결정 생성 단계 종료.")

    # AI 응답 파싱
    def parse_ai_response(response_text):
        try:
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                decision = parsed_json.get('decision')
                percentage = parsed_json.get('percentage')
                reason = parsed_json.get('reason')
                return {'decision': decision, 'percentage': percentage, 'reason': reason}
            else:
                logger.error("AI 응답에서 JSON을 찾을 수 없습니다.")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return None

    parsed_response = parse_ai_response(response_text)
    if not parsed_response:
        logger.error("AI 응답 파싱 실패.")
        return

    decision = parsed_response.get('decision')
    percentage = parsed_response.get('percentage')
    reason = parsed_response.get('reason')

    if not decision or reason is None:
        logger.error("AI 응답에 불완전한 데이터가 포함되어 있습니다.")
        return

    logger.info(f"AI Decision: {decision.upper()}")
    logger.info(f"Percentage: {percentage}")
    logger.info(f"Decision Reason: {reason}")

    order_executed = False

    # 거래 실행
    try:
        if decision.lower() == "buy":
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
                    logger.error(f"Buy order 실패: {order.get('ret_msg', 'No ret_msg')}")
            else:
                logger.warning("Buy Order Failed: Insufficient USDT (less than 5 USDT)")
        elif decision.lower() == "sell":
            try:
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
                                logger.error(f"Sell order 실패: {order.get('ret_msg', 'No ret_msg')}")
                        else:
                            logger.warning("Sell Order Failed: Calculated sell size is zero.")
                    else:
                        logger.info("판매할 포지션이 없습니다.")
                else:
                    logger.error(f"포지션 조회 실패 또는 포지션 없음: {position.get('ret_msg', 'No ret_msg')}")
            except Exception as e:
                logger.error(f"Sell order 실행 중 오류: {e}")
        elif decision.lower() == "hold":
            logger.info("Decision is to hold. No action taken.")
        else:
            logger.error("AI로부터 유효하지 않은 결정이 전달되었습니다.")
            return
    except Exception as e:
        logger.error(f"거래 실행 중 예상치 못한 오류 발생: {e}")
    else:
        logger.info("거래 실행 과정 완료.")
    finally:
        logger.debug("거래 실행 단계 종료.")

    # 거래 실행 여부와 관계없이 현재 잔고 조회 및 기록
    try:
        time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
        balance_data = get_account_balance()
        if balance_data:
            usdt_balance = balance_data['available_to_withdraw']
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
        logger.error(f"잔고 조회 및 거래 기록 저장 중 오류 발생: {e}")
    else:
        logger.info("잔고 조회 및 거래 기록 저장 완료.")
    finally:
        logger.debug("잔고 조회 및 기록 단계 종료.")

except Exception as e:
    logger.critical(f"AI 트레이딩 로직 전체 중 예상치 못한 오류 발생: {e}")
finally:
    logger.info("AI 트레이딩 로직 종료.")

# Scheduler setup
def run_scheduler():
    try:
        schedule.every().hour.do(ai_trading)
        logger.info("Scheduler started: ai_trading will run every hour.")
    except Exception as e:
        logger.critical(f"스케줄러 설정 오류: {e}")
        raise
    else:
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"스케줄러 실행 중 오류 발생: {e}")
            time.sleep(1)
    finally:
        logger.debug("run_scheduler 함수 종료.")

# Main function
def main():
    try:
        # Initialize MongoDB and Bybit connections
        init_db()
        setup_bybit()
    except Exception as e:
        logger.critical(f"초기화 중 오류 발생: {e}")
        return
    else:
        logger.info("초기화 과정 완료.")
    finally:
        logger.debug("초기화 단계 종료.")

    try:
        # Run ai_trading once at startup
        logger.info("Running initial ai_trading execution.")
        ai_trading()
    except Exception as e:
        logger.error(f"초기 ai_trading 실행 중 오류 발생: {e}")
    else:
        logger.info("초기 ai_trading 실행 완료.")
    finally:
        logger.debug("초기 ai_trading 단계 종료.")

    try:
        # Start the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    except Exception as e:
        logger.critical(f"스케줄러 스레드 시작 중 오류 발생: {e}")
        return
    else:
        logger.info("스케줄러 스레드 시작 완료.")
    finally:
        logger.debug("스케줄러 스레드 시작 단계 종료.")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 트레이딩 봇이 중지되었습니다.")
    except Exception as e:
        logger.critical(f"메인 루프 중 예상치 못한 오류 발생: {e}")
    finally:
        logger.debug("메인 함수 종료.")

if __name__ == "__main__":
    main()
