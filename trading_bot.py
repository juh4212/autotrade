import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import time
import requests
import logging
import sqlite3
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

# SQLite 데이터베이스 초기화 함수 - 거래 내역을 저장할 테이블을 생성
def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  decision TEXT,
                  percentage INTEGER,
                  reason TEXT,
                  btc_balance REAL,
                  krw_balance REAL,
                  btc_avg_buy_price REAL,
                  btc_krw_price REAL,
                  reflection TEXT)''')
    conn.commit()
    return conn

# 거래 기록을 DB에 저장하는 함수
def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection))
    conn.commit()

# 최근 투자 기록 조회
def get_recent_trades(conn, days=7):
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    return pd.DataFrame.from_records(data=c.fetchall(), columns=columns)

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0  # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (KRW + BTC * 현재 가격)
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    # 최종 잔고 계산
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    return (final_balance - initial_balance) / initial_balance * 100

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API 키가 없거나 유효하지 않습니다.")
        return None

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
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
    )

    try:
        response_content = response.choices[0].message.content
        return response_content
    except (IndexError, AttributeError) as e:
        logger.error(f"응답 내용 추출 중 오류 발생: {e}")
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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API 키가 없거나 유효하지 않습니다.")
        return None
    try:
        # 데이터베이스 연결
        with sqlite3.connect('bitcoin_trades.db') as conn:
            # 최근 거래 내역 가져오기
            recent_trades = get_recent_trades(conn)
            
            # 현재 시장 데이터 수집 (기존 코드에서 가져온 데이터 사용)
            current_market_data = {
                "orderbook": orderbook,
                "daily_ohlcv": df_daily_recent.to_dict(),
                "hourly_ohlcv": df_hourly_recent.to_dict()
            }
            
            # 반성 및 개선 내용 생성
            reflection = generate_reflection(recent_trades, current_market_data)
            
            # AI 모델에 반성 내용 제공
            # Few-shot prompting으로 JSON 예시 추가
            examples = """
예시 응답 1:
{
  "decision": "매수",
  "percentage": 50,
  "reason": "현재 시장 지표와 긍정적인 추세에 따라 투자하기에 좋은 기회입니다."
}

예시 응답 2:
{
  "decision": "매도",
  "percentage": 30,
  "reason": "시장에 부정적인 추세가 관찰되어 보유 자산의 일부를 매도하는 것이 좋습니다."
}

예시 응답 3:
{
  "decision": "보유",
  "percentage": 0,
  "reason": "시장 지표가 중립적이므로 더 명확한 신호를 기다리는 것이 좋습니다."
}
"""

            response = client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "user",
                        "content": f"""당신은 비트코인 투자 전문가입니다. 이 분석은 매 4시간마다 수행됩니다. 제공된 데이터를 분석하고 현재 시점에서 매수, 매도 또는 보유 중 어떤 행동을 취해야 하는지 결정하세요. 분석 시 다음 사항을 고려하세요:

- 기술적 지표와 시장 데이터
- 전체적인 시장 심리
- 최근 거래 성과와 반성 내용

최근 거래 반성 내용:
{reflection}

분석에 기반하여 결정을 내리고 그 이유를 제공하세요.

다음의 JSON 형식으로 응답을 제공하세요:

{examples}

percentage는 매수/매도 결정 시 1에서 100 사이의 정수여야 하며, 보유 결정 시 정확히 0이어야 합니다.
percentage는 분석된 데이터에 기반한 결정의 강도를 반영해야 합니다.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""현재 투자 상태: {json.dumps(filtered_balances)}
호가 정보: {json.dumps(orderbook)}
일간 OHLCV 데이터와 지표 (최근 60일): {df_daily_recent.to_json()}
시간별 OHLCV 데이터와 지표 (최근 48시간): {df_hourly_recent.to_json()}
"""
                    }
                ]
            )

            response_text = response.choices[0].message.content

            # AI 응답 파싱
            def parse_ai_response(response_text):
                try:
                    # JSON 부분만 추출
                    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        # JSON 파싱
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
                logger.error("AI 응답 파싱에 실패하였습니다.")
                return

            decision = parsed_response.get('decision')
            percentage = parsed_response.get('percentage')
            reason = parsed_response.get('reason')

            if not decision or reason is None:
                logger.error("AI 응답에 불완전한 데이터가 있습니다.")
                return

            logger.info(f"AI 결정: {decision.upper()}")
            logger.info(f"비율: {percentage}")
            logger.info(f"결정 이유: {reason}")

            order_executed = False

            if decision == "매수":
                my_krw = upbit.get_balance("KRW")
                if my_krw is None:
                    logger.error("KRW 잔고 조회에 실패하였습니다.")
                    return
                buy_amount = my_krw * (percentage / 100) * 0.9995  # 수수료 고려
                if buy_amount > 5000:
                    logger.info(f"매수 주문 실행: 보유 KRW의 {percentage}%")
                    try:
                        order = upbit.buy_market_order("KRW-BTC", buy_amount)
                        if order:
                            logger.info(f"매수 주문이 성공적으로 실행되었습니다: {order}")
                            order_executed = True
                        else:
                            logger.error("매수 주문에 실패하였습니다.")
                    except Exception as e:
                        logger.error(f"매수 주문 실행 중 오류 발생: {e}")
                else:
                    logger.warning("매수 주문 실패: 잔고가 부족합니다 (5,000 KRW 미만)")
            elif decision == "매도":
                my_btc = upbit.get_balance("KRW-BTC")
                if my_btc is None:
                    logger.error("BTC 잔고 조회에 실패하였습니다.")
                    return
                sell_amount = my_btc * (percentage / 100)
                current_price = pyupbit.get_current_price("KRW-BTC")
                if sell_amount * current_price > 5000:
                    logger.info(f"매도 주문 실행: 보유 BTC의 {percentage}%")
                    try:
                        order = upbit.sell_market_order("KRW-BTC", sell_amount)
                        if order:
                            order_executed = True
                        else:
                            logger.error("매도 주문에 실패하였습니다.")
                    except Exception as e:
                        logger.error(f"매도 주문 실행 중 오류 발생: {e}")
                else:
                    logger.warning("매도 주문 실패: 잔고가 부족합니다 (5,000 KRW 미만)")
            elif decision == "보유":
                logger.info("결정은 보유입니다. 아무 행동도 취하지 않습니다.")
            else:
                logger.error("AI로부터 유효하지 않은 결정을 받았습니다.")
                return

            # 거래 실행 여부와 관계없이 현재 잔고 조회
            time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
            balances = upbit.get_balances()
            btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
            krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
            btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
            current_btc_price = pyupbit.get_current_price("KRW-BTC")

            # 거래 기록을 DB에 저장하기
            log_trade(conn, decision, percentage if order_executed else 0, reason, 
                      btc_balance, krw_balance, btc_avg_buy_price, current_btc_price, reflection)
    except sqlite3.Error as e:
        logger.error(f"데이터베이스 연결 오류: {e}")
        return

if __name__ == "__main__":
    # 데이터베이스 초기화
    init_db()

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
