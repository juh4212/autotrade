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
from pybit.usdt_perpetual import HTTP  # 선물 거래용 HTTP 클래스 임포트
import openai
import ta
from ta.utils import dropna
from pymongo import MongoClient  # MongoDB 클라이언트 추가
from urllib.parse import quote_plus  # 비밀번호 URL 인코딩을 위한 모듈
# from selenium import webdriver  # 이미지 캡처가 필요하다면 주석 해제

# 로깅 설정 - 로그 레벨을 DEBUG로 설정하고, 콘솔에만 출력
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bybit 객체 생성
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
if not api_key or not api_secret:
    logger.error("API keys not found. Please check your environment variables.")
    raise ValueError("Missing API keys. Please check your environment variables.")

logger.debug("Bybit API 키가 성공적으로 로드되었습니다.")

# Bybit REST API 세션 생성 (USDT 선물)
try:
    session = HTTP(
        endpoint="https://api.bybit.com",  # 실제 거래를 위한 엔드포인트
        api_key=api_key,
        api_secret=api_secret
    )
    logger.debug("Bybit REST API 세션이 성공적으로 생성되었습니다.")
except Exception as e:
    logger.exception(f"Bybit REST API 세션 생성 실패: {e}")
    raise

# MongoDB 연결 설정
def init_db():
    logger.debug("init_db 함수 시작")
    # 환경 변수에서 MongoDB 비밀번호 가져오기
    db_password = os.getenv("MONGODB_PASSWORD")
    if not db_password:
        logger.error("MongoDB password not found. Please set the MONGODB_PASSWORD environment variable.")
        raise ValueError("Missing MongoDB password.")

    # 비밀번호를 URL 인코딩
    encoded_password = quote_plus(db_password)

    # MongoDB Atlas 연결 문자열 구성
    mongo_uri = f"mongodb+srv://juh4212:{encoded_password}@cluster1.tbzg2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"

    try:
        # MongoDB 클라이언트 생성
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # 서버 정보 조회로 연결 확인
        client.server_info()
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        logger.debug("MongoDB에 성공적으로 연결되었습니다.")
        return trades_collection
    except Exception as e:
        logger.exception(f"MongoDB 연결 실패: {e}")
        raise

# 거래 기록을 DB에 저장하는 함수
def log_trade(trades_collection, decision, percentage, reason, btc_balance,
              usdt_balance, btc_avg_buy_price, btc_usdt_price, reflection=''):
    logger.debug("log_trade 함수 시작")
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
    try:
        trades_collection.insert_one(trade)
        logger.debug("거래 기록이 성공적으로 DB에 저장되었습니다.")
    except Exception as e:
        logger.exception(f"거래 기록 DB 저장 실패: {e}")

# 최근 투자 기록 조회
def get_recent_trades(trades_collection, days=7):
    logger.debug(f"get_recent_trades 함수 시작 - 최근 {days}일간의 거래 내역 조회")
    seven_days_ago = datetime.now() - timedelta(days=days)
    try:
        cursor = trades_collection.find({"timestamp": {"$gte": seven_days_ago}}).sort("timestamp", -1)
        trades = list(cursor)
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            logger.debug(f"최근 거래 내역 조회 성공 - 총 {len(trades_df)}건")
        else:
            logger.debug("최근 거래 내역이 없습니다.")
        return trades_df
    except Exception as e:
        logger.exception(f"최근 거래 내역 조회 실패: {e}")
        return pd.DataFrame()

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    logger.debug("calculate_performance 함수 시작")
    if trades_df.empty:
        logger.debug("거래 기록이 없어 퍼포먼스를 0%로 설정합니다.")
        return 0  # 기록이 없을 경우 0%로 설정
    try:
        # 초기 잔고 계산 (USDT + BTC * 당시 가격)
        initial_balance = trades_df.iloc[-1]['usdt_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_usdt_price']
        # 최종 잔고 계산
        final_balance = trades_df.iloc[0]['usdt_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_usdt_price']
        performance = (final_balance - initial_balance) / initial_balance * 100
        logger.debug(f"퍼포먼스 계산 완료: {performance:.2f}%")
        return performance
    except Exception as e:
        logger.exception(f"퍼포먼스 계산 실패: {e}")
        return 0

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    logger.debug("generate_reflection 함수 시작")
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key가 누락되었거나 유효하지 않습니다.")
        return None

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 모델 변경: o1-preview -> gpt-4
            messages=[
                {
                    "role": "system",
                    "content": "당신은 비트코인 선물 트레이딩 전문가입니다. 제공된 데이터를 분석하고 현재 시점에서 최선의 행동을 결정하세요."
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
            ],
            timeout=60  # 시간 초과 설정 (초)
        )
        logger.debug("OpenAI API 호출 성공")
        response_content = response.choices[0].message.content
        logger.debug(f"AI 응답 내용: {response_content}")
        return response_content
    except Exception as e:
        logger.exception(f"OpenAI API 호출 실패: {e}")
        return None

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df):
    logger.debug("add_indicators 함수 시작")
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

        logger.debug("보조 지표 추가 완료")
        return df
    except Exception as e:
        logger.exception(f"보조 지표 추가 실패: {e}")
        return df

# 공포 탐욕 지수 조회
def get_fear_and_greed_index():
    logger.debug("get_fear_and_greed_index 함수 시작")
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"공포 탐욕 지수 데이터: {data['data'][0]}")
        return data['data'][0]
    except requests.exceptions.RequestException as e:
        logger.exception(f"공포 탐욕 지수 조회 실패: {e}")
        return None

# 가격 데이터 가져오기 함수 (Bybit용)
def get_ohlcv(symbol, interval, limit):
    logger.debug(f"get_ohlcv 함수 시작 - symbol: {symbol}, interval: {interval}, limit: {limit}")
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["retCode"] != 0:
            logger.error(f"OHLCV 데이터 조회 오류: {data['retMsg']}")
            return None
        records = data["result"]["list"]
        df = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        logger.debug(f"OHLCV 데이터 조회 성공 - 총 {len(df)}건")
        return df
    except requests.RequestException as e:
        logger.exception(f"OHLCV 데이터 조회 실패: {e}")
        return None

### 메인 AI 트레이딩 로직
def ai_trading():
    logger.debug("ai_trading 함수 시작")
    global session
    ### 데이터 가져오기
    # 1. 현재 포지션 조회 (선물 거래에 맞게 수정)
    try:
        logger.debug("현재 포지션 조회 시도")
        positions = session.my_position(symbol="BTCUSDT")['result']
        # 포지션 정보 파싱
        long_position = next((p for p in positions if p['side'] == 'Buy'), None)
        short_position = next((p for p in positions if p['side'] == 'Sell'), None)
        logger.debug(f"롱 포지션: {long_position}, 숏 포지션: {short_position}")
    except Exception as e:
        logger.exception(f"포지션 조회 실패: {e}")
        return

    # 2. 현재 잔고 조회 (선물 지갑 잔고)
    try:
        logger.debug("현재 잔고 조회 시도")
        wallet_balance = session.get_wallet_balance()['result']['USDT']['available_balance']
        usdt_balance = float(wallet_balance)
        logger.debug(f"USDT 잔고: {usdt_balance}")
    except Exception as e:
        logger.exception(f"잔고 조회 실패: {e}")
        return

    # 3. 오더북(호가 데이터) 조회
    try:
        logger.debug("오더북 조회 시도")
        orderbook_response = session.orderbook(symbol="BTCUSDT")
        orderbook = orderbook_response['result']
        logger.debug(f"오더북 데이터: {orderbook}")
    except Exception as e:
        logger.exception(f"오더북 조회 실패: {e}")
        orderbook = None

    # 4. 차트 데이터 조회 및 보조지표 추가
    try:
        logger.debug("차트 데이터 조회 시도 - 일일 데이터")
        df_daily = get_ohlcv("BTCUSDT", interval="D", limit=180)
        if df_daily is None:
            logger.error("일일 OHLCV 데이터 조회 실패")
            return
        df_daily = dropna(df_daily)
        df_daily = add_indicators(df_daily)
        
        logger.debug("차트 데이터 조회 시도 - 시간별 데이터")
        df_hourly = get_ohlcv("BTCUSDT", interval="60", limit=168)  # 7 days of hourly data
        if df_hourly is None:
            logger.error("시간별 OHLCV 데이터 조회 실패")
            return
        df_hourly = dropna(df_hourly)
        df_hourly = add_indicators(df_hourly)
        
        # 최근 데이터만 사용하도록 설정 (메모리 절약)
        df_daily_recent = df_daily.tail(60)
        df_hourly_recent = df_hourly.tail(48)
        logger.debug(f"최근 일일 데이터: {df_daily_recent.shape[0]}건, 최근 시간별 데이터: {df_hourly_recent.shape[0]}건")
    except Exception as e:
        logger.exception(f"차트 데이터 조회 또는 보조지표 추가 실패: {e}")
        return

    # 5. 공포 탐욕 지수 가져오기
    try:
        logger.debug("공포 탐욕 지수 조회 시도")
        fear_greed_index = get_fear_and_greed_index()
        logger.debug(f"공포 탐욕 지수: {fear_greed_index}")
    except Exception as e:
        logger.exception(f"공포 탐욕 지수 조회 실패: {e}")
        fear_greed_index = None

    # 6. 뉴스 데이터 가져오기 (필요 시 활성화)
    # image_data = capture_chart()  # 이미지 캡처 함수 호출 (필요 시 활성화)

    ### AI에게 데이터 제공하고 판단 받기
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key가 누락되었거나 유효하지 않습니다.")
        return None
    try:
        # 데이터베이스 연결
        logger.debug("데이터베이스 연결 시도")
        trades_collection = init_db()

        # 최근 거래 내역 가져오기
        logger.debug("최근 거래 내역 조회 시도")
        recent_trades = get_recent_trades(trades_collection)

        # 현재 시장 데이터 수집
        current_market_data = {
            "fear_greed_index": fear_greed_index,
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.describe().to_dict(),  # 요약 통계로 대체
            "hourly_ohlcv": df_hourly_recent.describe().to_dict()  # 요약 통계로 대체
            # "chart_image": encoded_image  # 이미지 데이터 포함 (필요 시 추가)
        }
        logger.debug(f"현재 시장 데이터: {current_market_data}")

        # 반성 및 개선 내용 생성
        logger.debug("반성 및 개선 내용 생성 시도")
        reflection = generate_reflection(recent_trades, current_market_data)
        logger.debug(f"생성된 반성 내용: {reflection}")

        # AI 모델에 반성 내용 제공
        # Few-shot prompting으로 JSON 예시 추가
        examples = """
Example Response 1:
{
  "decision": "open_long",
  "percentage": 50,
  "leverage": 5,
  "reason": "기술 지표와 시장 상황을 고려할 때, 롱 포지션을 여는 것이 유리합니다."
}

Example Response 2:
{
  "decision": "close_long",
  "percentage": 100,
  "reason": "현재 롱 포지션을 보유 중이며, 시장 지표가 반전을 시사하여 롱 포지션을 청산하는 것이 바람직합니다."
}

Example Response 3:
{
  "decision": "open_short",
  "percentage": 30,
  "leverage": 3,
  "reason": "시장 트렌드가 부정적으로 전개되고 공포 지수가 높아 숏 포지션을 여는 것이 유리합니다."
}

Example Response 4:
{
  "decision": "close_short",
  "percentage": 100,
  "reason": "현재 숏 포지션을 보유 중이며, 시장 지표가 상승 추세를 시사하여 숏 포지션을 청산하는 것이 바람직합니다."
}

Example Response 5:
{
  "decision": "hold",
  "percentage": 0,
  "reason": "시장 지표가 중립적이어서 명확한 신호가 없으므로 관망하는 것이 좋습니다."
}
"""

        # 메시지 구성
        messages = [
            {
                "role": "system",
                "content": "당신은 비트코인 선물 트레이딩 전문가입니다. 제공된 데이터를 분석하고 현재 시점에서 최선의 행동을 결정하세요."
            },
            {
                "role": "user",
                "content": f"""
Possible decisions:
- "open_long": 롱 포지션 열기
- "close_long": 롱 포지션 청산
- "open_short": 숏 포지션 열기
- "close_short": 숏 포지션 청산
- "hold": 관망하기

레버리지는 1에서 10 사이의 정수로 설정하며, 새로운 포지션을 열 때만 포함합니다.

다음과 같은 JSON 형식으로 응답하세요:

{examples}

투자 판단을 내려주세요.
"""
            },
            {
                "role": "user",
                "content": f"""
현재 포지션: 롱 - {long_position}, 숏 - {short_position}
사용 가능한 USDT 잔고: {usdt_balance}
오더북 요약: {json.dumps(orderbook)}
일일 OHLCV 요약: {json.dumps(df_daily_recent.describe().to_dict())}
시간별 OHLCV 요약: {json.dumps(df_hourly_recent.describe().to_dict())}
공포 탐욕 지수: {json.dumps(fear_greed_index)}
"""
            }
        ]

        logger.debug("OpenAI API 호출 준비 완료")

        response = openai.ChatCompletion.create(
            model="gpt-4",  # 모델 변경: o1-preview -> gpt-4
            messages=messages,
            timeout=60  # 시간 초과 설정 (초)
        )
        logger.debug("OpenAI API 호출 성공")

        response_text = response.choices[0].message.content
        logger.debug(f"AI 응답 내용: {response_text}")

        # AI 응답 파싱
        def parse_ai_response(response_text):
            logger.debug("parse_ai_response 함수 시작")
            try:
                # JSON 부분 추출
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    decision = parsed_json.get('decision')
                    percentage = parsed_json.get('percentage')
                    leverage = parsed_json.get('leverage')
                    reason = parsed_json.get('reason')
                    logger.debug(f"파싱된 JSON: {parsed_json}")
                    return {'decision': decision, 'percentage': percentage, 'leverage': leverage, 'reason': reason}
                else:
                    # JSON 형식이 아닐 경우 텍스트에서 정보 추출
                    logger.debug("응답이 JSON 형식이 아님, 텍스트에서 정보 추출 시도")
                    decision_match = re.search(r'Decision:\s*(\w+)', response_text, re.IGNORECASE)
                    percentage_match = re.search(r'Percentage:\s*(\d+)', response_text, re.IGNORECASE)
                    leverage_match = re.search(r'Leverage:\s*(\d+)', response_text, re.IGNORECASE)
                    reason_match = re.search(r'Reason:\s*(.*)', response_text, re.IGNORECASE)
                    decision = decision_match.group(1) if decision_match else None
                    percentage = int(percentage_match.group(1)) if percentage_match else None
                    leverage = int(leverage_match.group(1)) if leverage_match else None
                    reason = reason_match.group(1).strip() if reason_match else None
                    parsed_data = {'decision': decision, 'percentage': percentage, 'leverage': leverage, 'reason': reason}
                    logger.debug(f"파싱된 텍스트 데이터: {parsed_data}")
                    return parsed_data
            except Exception as e:
                logger.exception(f"AI 응답 파싱 실패: {e}")
                return None

        parsed_response = parse_ai_response(response_text)
        if not parsed_response:
            logger.error("AI 응답을 파싱할 수 없습니다.")
            return

        decision = parsed_response.get('decision')
        percentage = parsed_response.get('percentage')
        leverage = parsed_response.get('leverage')
        reason = parsed_response.get('reason')

        if not decision or reason is None or percentage is None:
            logger.error("AI 응답에 불완전한 데이터가 포함되어 있습니다.")
            return

        logger.info(f"AI Decision: {decision.upper()}")
        logger.info(f"Percentage: {percentage}")
        if leverage:
            logger.info(f"Leverage: {leverage}")
        logger.info(f"Decision Reason: {reason}")

        order_executed = False

        # 현재 가격 가져오기
        try:
            logger.debug("현재 가격 데이터 조회 시도")
            current_price_data = session.latest_information_for_symbol(symbol="BTCUSDT")
            current_price = float(current_price_data['result'][0]['last_price'])
            logger.debug(f"현재 BTC 가격: {current_price}")
        except Exception as e:
            logger.exception(f"현재 가격 데이터 조회 실패: {e}")
            return

        # 주문 실행
        try:
            if decision == "open_long":
                # 레버리지 확인
                if leverage is None:
                    logger.error("레버리지가 필요합니다. 포지션을 여는 데 실패했습니다.")
                    return
                leverage = max(1, min(int(leverage), 10))
                session.set_leverage(symbol="BTCUSDT", buy_leverage=leverage, sell_leverage=leverage)
                position_size = usdt_balance * (int(percentage) / 100) * 0.9995  # 수수료 고려
                if position_size > 10:  # 최소 거래 금액은 거래소에 따라 다를 수 있음
                    logger.info(f"롱 포지션 주문 시도: {percentage}%의 USDT와 {leverage}x 레버리지")
                    order_qty = round((position_size * leverage) / current_price, 6)  # 레버리지 적용
                    try:
                        order = session.place_active_order(
                            symbol="BTCUSDT",
                            side="Buy",
                            order_type="Market",
                            qty=order_qty,
                            time_in_force="GoodTillCancel",
                            reduce_only=False,
                            close_on_trigger=False
                        )
                        if order['ret_code'] == 0:
                            logger.info(f"롱 포지션 주문 성공: {order}")
                            order_executed = True
                        else:
                            logger.error(f"롱 포지션 주문 실패: {order['ret_msg']}")
                    except Exception as e:
                        logger.exception(f"롱 포지션 주문 중 오류 발생: {e}")
                else:
                    logger.warning("롱 주문 실패: USDT 잔고가 부족합니다.")
            elif decision == "close_long":
                # 롱 포지션 청산 로직
                if long_position and float(long_position['size']) > 0:
                    logger.info("롱 포지션 청산 시도")
                    order_qty = float(long_position['size'])
                    try:
                        order = session.place_active_order(
                            symbol="BTCUSDT",
                            side="Sell",
                            order_type="Market",
                            qty=order_qty,
                            time_in_force="GoodTillCancel",
                            reduce_only=True,
                            close_on_trigger=False
                        )
                        if order['ret_code'] == 0:
                            logger.info(f"롱 포지션 청산 성공: {order}")
                            order_executed = True
                        else:
                            logger.error(f"롱 포지션 청산 실패: {order['ret_msg']}")
                    except Exception as e:
                        logger.exception(f"롱 포지션 청산 중 오류 발생: {e}")
                else:
                    logger.info("청산할 롱 포지션이 없습니다.")
            elif decision == "open_short":
                # 레버리지 확인
                if leverage is None:
                    logger.error("레버리지가 필요합니다. 포지션을 여는 데 실패했습니다.")
                    return
                leverage = max(1, min(int(leverage), 10))
                session.set_leverage(symbol="BTCUSDT", buy_leverage=leverage, sell_leverage=leverage)
                position_size = usdt_balance * (int(percentage) / 100) * 0.9995  # 수수료 고려
                if position_size > 10:
                    logger.info(f"숏 포지션 주문 시도: {percentage}%의 USDT와 {leverage}x 레버리지")
                    order_qty = round((position_size * leverage) / current_price, 6)
                    try:
                        order = session.place_active_order(
                            symbol="BTCUSDT",
                            side="Sell",
                            order_type="Market",
                            qty=order_qty,
                            time_in_force="GoodTillCancel",
                            reduce_only=False,
                            close_on_trigger=False
                        )
                        if order['ret_code'] == 0:
                            logger.info(f"숏 포지션 주문 성공: {order}")
                            order_executed = True
                        else:
                            logger.error(f"숏 포지션 주문 실패: {order['ret_msg']}")
                    except Exception as e:
                        logger.exception(f"숏 포지션 주문 중 오류 발생: {e}")
                else:
                    logger.warning("숏 주문 실패: USDT 잔고가 부족합니다.")
            elif decision == "close_short":
                # 숏 포지션 청산 로직
                if short_position and float(short_position['size']) > 0:
                    logger.info("숏 포지션 청산 시도")
                    order_qty = float(short_position['size'])
                    try:
                        order = session.place_active_order(
                            symbol="BTCUSDT",
                            side="Buy",
                            order_type="Market",
                            qty=order_qty,
                            time_in_force="GoodTillCancel",
                            reduce_only=True,
                            close_on_trigger=False
                        )
                        if order['ret_code'] == 0:
                            logger.info(f"숏 포지션 청산 성공: {order}")
                            order_executed = True
                        else:
                            logger.error(f"숏 포지션 청산 실패: {order['ret_msg']}")
                    except Exception as e:
                        logger.exception(f"숏 포지션 청산 중 오류 발생: {e}")
                else:
                    logger.info("청산할 숏 포지션이 없습니다.")
            elif decision == "hold":
                logger.info("결정: 관망. 아무 조치도 취하지 않습니다.")
            else:
                logger.error("AI로부터 유효하지 않은 결정을 받았습니다.")
                return

            # 거래 실행 여부와 관계없이 현재 잔고 및 포지션 조회
            logger.debug("거래 후 잔고 및 포지션 조회 시도")
            time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
            try:
                positions = session.my_position(symbol="BTCUSDT")['result']
                long_position = next((p for p in positions if p['side'] == 'Buy'), None)
                short_position = next((p for p in positions if p['side'] == 'Sell'), None)
                wallet_balance = session.get_wallet_balance()['result']['USDT']['available_balance']
                usdt_balance = float(wallet_balance)
                btc_balance = (float(long_position['size']) if long_position else 0) - \
                              (float(short_position['size']) if short_position else 0)
                btc_avg_buy_price = float(long_position['entry_price']) if long_position else None
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
                logger.exception(f"거래 후 잔고 및 포지션 조회 실패: {e}")
        except Exception as e:
            logger.exception(f"주문 실행 중 오류 발생: {e}")
            return

    except Exception as e:
        logger.exception(f"AI 트레이딩 로직 중 오류 발생: {e}")
        return

    logger.debug("ai_trading 함수 종료")

if __name__ == "__main__":
    logger.debug("메인 스크립트 시작")
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
            logger.debug("트레이딩 작업 시작")
            ai_trading()
        except Exception as e:
            logger.exception(f"트레이딩 작업 중 오류 발생: {e}")
        finally:
            trading_in_progress = False
            logger.debug("트레이딩 작업 종료")

# 매 1분마다 실행 (테스트 용도)
schedule.every(1).minutes.do(job)

# 원래 스케줄은 주석 처리하거나 제거
# schedule.every().day.at("00:00").do(job)
# schedule.every().day.at("04:00").do(job)
# schedule.every().day.at("08:00").do(job)
# schedule.every().day.at("12:00").do(job)
# schedule.every().day.at("16:00").do(job)
# schedule.every().day.at("20:00").do(job)


    logger.debug("스케줄러 설정 완료")

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.exception(f"스케줄러 루프 중 오류 발생: {e}")
