import os
import logging
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP  # Bybit v5 API를 사용 중임을 가정
import pandas as pd
import ta
import openai

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
            logger.critical("Bybit API 키가 설정되지 않았습니다.")
            raise ValueError("Bybit API 키가 누락되었습니다.")
        bybit = HTTP(api_key=api_key, api_secret=api_secret)
        logger.info("Bybit API 연결 완료!")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

# Bybit 계좌 잔고 조회
def get_account_balance(bybit):
    try:
        wallet_balance = bybit.get_wallet_balance(accountType="CONTRACT")
        logger.info("Bybit API 응답 데이터: %s", wallet_balance)  # 전체 응답 데이터 출력
        if wallet_balance['retCode'] == 0 and 'result' in wallet_balance:
            account_list = wallet_balance['result'].get('list', [])
            if account_list:
                account_info = account_list[0]
                coin_balances = account_info.get('coin', [])
                
                usdt_balance = next((coin for coin in coin_balances if coin['coin'] == 'USDT'), None)
                if usdt_balance:
                    equity = float(usdt_balance.get('equity', 0))
                    available_to_withdraw = float(usdt_balance.get('availableToWithdraw', 0))
                    logger.info(f"USDT 전체 자산: {equity}, 사용 가능한 자산: {available_to_withdraw}")
                    return {
                        "equity": equity,
                        "available_to_withdraw": available_to_withdraw
                    }
                else:
                    logger.error("USDT 잔고 데이터를 찾을 수 없습니다.")
                    return None
            else:
                logger.error("계정 리스트가 비어 있습니다.")
                return None
        else:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return None
    except Exception as e:
        logger.error(f"Bybit 잔고 조회 오류: {e}")
        return None

# MongoDB에 잔고 기록
def log_balance_to_mongodb(collection, balance_data):
    balance_record = {
        "timestamp": datetime.utcnow(),
        "balance_data": balance_data
    }
    try:
        collection.insert_one(balance_record)
        logger.info("계좌 잔고가 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"MongoDB에 계좌 잔고 저장 오류: {e}")

# 거래 기록을 MongoDB에 저장하는 함수
def log_trade(collection, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    trade_record = {
        "timestamp": datetime.utcnow(),
        "decision": decision,
        "percentage": percentage,
        "reason": reason,
        "btc_balance": btc_balance,
        "krw_balance": krw_balance,
        "btc_avg_buy_price": btc_avg_buy_price,
        "btc_krw_price": btc_krw_price,
        "reflection": reflection
    }
    try:
        collection.insert_one(trade_record)
        logger.info("거래 기록이 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"MongoDB에 거래 기록 저장 오류: {e}")

# 최근 투자 기록 조회
def get_recent_trades(collection, days=7):
    try:
        seven_days_ago = datetime.utcnow() - timedelta(days=days)
        cursor = collection.find({"timestamp": {"$gte": seven_days_ago}}).sort("timestamp", -1)
        trades = list(cursor)
        if trades:
            df = pd.DataFrame(trades)
            return df
        else:
            logger.info("최근 거래 기록이 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"최근 거래 기록 조회 오류: {e}")
        return pd.DataFrame()

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    try:
        # 초기 잔고 계산 (KRW + BTC * 초기 BTC/KRW 가격)
        initial_trade = trades_df.iloc[-1]
        initial_balance = initial_trade['krw_balance'] + initial_trade['btc_balance'] * initial_trade['btc_krw_price']
        # 최종 잔고 계산
        final_trade = trades_df.iloc[0]
        final_balance = final_trade['krw_balance'] + final_trade['btc_balance'] * final_trade['btc_krw_price']
        performance = (final_balance - initial_balance) / initial_balance * 100
        logger.info(f"퍼포먼스 계산: 초기 잔고={initial_balance}, 최종 잔고={final_balance}, 퍼포먼스={performance:.2f}%")
        return performance
    except Exception as e:
        logger.error(f"퍼포먼스 계산 오류: {e}")
        return 0

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    try:
        performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.error("OpenAI API key가 누락되었거나 유효하지 않습니다.")
            return None

        # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
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
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        try:
            response_content = response.choices[0].message.content.strip()
            logger.info("AI 반성 생성 완료.")
            return response_content
        except (IndexError, AttributeError) as e:
            logger.error(f"응답 내용 추출 오류: {e}")
            return None
    except Exception as e:
        logger.error(f"반성 생성 오류: {e}")
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

# 메인 AI 트레이딩 로직
def ai_trading(bybit, collection):
    try:
        # 1. 현재 투자 상태 조회
        wallet_balance = get_account_balance(bybit)
        if not wallet_balance:
            logger.error("잔고 조회 실패로 인해 AI 트레이딩을 중단합니다.")
            return

        filtered_balances = {
            "USDT": wallet_balance['equity'],
            "Available_USDT": wallet_balance['available_to_withdraw']
        }

        # 2. 오더북(호가 데이터) 조회
        orderbook = bybit.get_orderbook(symbol="BTCUSDT")
        logger.info("오더북 데이터 가져오기 완료.")

        # 3. 차트 데이터 조회 및 보조지표 추가
        # Bybit에서는 Kline 데이터를 가져와야 합니다. 예를 들어, 1일 및 1시간 간격
        klines_daily = bybit.get_kline(symbol="BTCUSDT", interval="D", limit=180)
        klines_hourly = bybit.get_kline(symbol="BTCUSDT", interval="60", limit=168)

        # 데이터프레임으로 변환
        df_daily = pd.DataFrame(klines_daily['result'])
        df_daily.rename(columns={
            'open_time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'turnover': 'turnover'
        }, inplace=True)
        df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='s')
        df_daily.set_index('timestamp', inplace=True)

        df_hourly = pd.DataFrame(klines_hourly['result'])
        df_hourly.rename(columns={
            'open_time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'turnover': 'turnover'
        }, inplace=True)
        df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'], unit='s')
        df_hourly.set_index('timestamp', inplace=True)

        # 보조 지표 추가
        df_daily = add_indicators(df_daily)
        df_hourly = add_indicators(df_hourly)

        # 최근 데이터만 사용하도록 설정 (메모리 절약)
        df_daily_recent = df_daily.tail(60)
        df_hourly_recent = df_hourly.tail(48)

        # 최근 거래 기록 조회
        recent_trades_df = get_recent_trades(collection, days=7)

        # 현재 시장 데이터 수집 (뉴스와 공포 지표는 제외)
        current_market_data = {
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.to_dict(orient='records'),
            "hourly_ohlcv": df_hourly_recent.to_dict(orient='records')
        }

        # 반성 및 개선 내용 생성
        reflection = generate_reflection(recent_trades_df, current_market_data)

        # AI에게 데이터 제공하고 판단 받기
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.error("OpenAI API key가 누락되었거나 유효하지 않습니다.")
            return

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
  "reason": "Due to negative trends in the market and declining indicators, it is advisable to reduce holdings."
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
                    "content": "You are an expert in Bitcoin investing. This analysis is performed every 4 hours. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:\n\n- Technical indicators and market data\n- Overall market sentiment\n- Recent trading performance and reflection\n\nRecent trading reflection:\n" + reflection
                },
                {
                    "role": "user",
                    "content": f"""Based on the following data, make a decision to buy, sell, or hold Bitcoin. Provide your response in the following JSON format:

{examples}

Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions. Your percentage should reflect the strength of your conviction in the decision based on the analyzed data.
"""
                },
                {
                    "role": "user",
                    "content": f"""Current investment status: {json.dumps(filtered_balances)}
Orderbook: {json.dumps(orderbook)}
Daily OHLCV with indicators (recent 60 days): {df_daily_recent.to_json(orient='records')}
Hourly OHLCV with indicators (recent 48 hours): {df_hourly_recent.to_json(orient='records')}
"""
                }
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        response_text = response.choices[0].message.content.strip()
        logger.info("AI 의사결정 생성 완료.")
        logger.info("AI 응답: %s", response_text)
