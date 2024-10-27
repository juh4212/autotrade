import os
import logging
import time
import json
import re
from datetime import datetime, timedelta
import requests
import pandas as pd
import openai
import ta
from ta.utils import dropna
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import hashlib
import hmac
from dotenv import load_dotenv
import schedule
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 환경 변수 로드 (.env 파일 사용 시)
load_dotenv()

# 테스트 모드 플래그 설정 (환경 변수로 관리 가능)
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

# 로깅 설정 - DEBUG 레벨로 설정하여 자세한 로그 기록
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 변경하여 모든 로그 기록
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔에 로그 출력
        RotatingFileHandler("logs/trading_bot.log", maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')  # 최대 5MB, 백업 5개
    ]
)
logger = logging.getLogger(__name__)

logger.info("트레이딩 봇 초기화 시작")

# Bybit V5 API 엔드포인트
BASE_URL = "https://api.bybit.com"

# API 키 및 시크릿
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    logger.error("API keys not found. Please check your environment variables.")
    raise ValueError("Missing API keys. Please check your environment variables.")

logger.info("Bybit API 키가 성공적으로 로드되었습니다.")
logger.debug(f"BYBIT_API_KEY: {API_KEY}, BYBIT_API_SECRET: {'***' if API_SECRET else 'None'}")

# 시그니처 생성 함수
def generate_signature(params, secret):
    """시그니처 생성"""
    try:
        # 시그니처 생성 시 'sign' 파라미터는 제외
        params_to_sign = {k: v for k, v in params.items() if k != 'sign'}
        ordered_params = '&'.join([f"{key}={params_to_sign[key]}" for key in sorted(params_to_sign)])
        signature = hmac.new(secret.encode(), ordered_params.encode(), hashlib.sha256).hexdigest()
        logger.debug(f"시그니처 생성: {signature} from params: {ordered_params}")
        return signature
    except Exception as e:
        logger.exception(f"시그니처 생성 중 오류 발생: {e}")
        raise

# 세션 생성 및 재시도 설정
def create_session():
    logger.debug("HTTP 세션 생성 시도")
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,  # 지수 백오프 시작 지연 시간 (초)
        status_forcelist=[503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    logger.debug("HTTP 세션 생성 완료")
    return session

# Bybit V5 API 호출 함수
def call_bybit_api(endpoint, method='GET', params=None, data=None, max_retries=5):
    """Bybit V5 API 호출 함수"""
    url = BASE_URL + endpoint
    headers = {
        "Content-Type": "application/json"
    }
    session = create_session()
    attempt = 0
    while attempt < max_retries:
        try:
            if method.upper() == 'GET':
                logger.debug(f"GET 요청: {url} with params: {params}")
                response = session.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                logger.debug(f"POST 요청: {url} with data: {data}")
                response = session.post(url, json=data, headers=headers, timeout=10)
            else:
                logger.error(f"지원되지 않는 HTTP 메서드: {method}")
                return None

            logger.debug(f"응답 상태 코드: {response.status_code}")
            logger.debug(f"응답 내용: {response.text}")

            if response.status_code == 503:
                logger.warning(f"503 오류 발생: {response.text}. 재시도 시도 {attempt + 1}/{max_retries}")
                attempt += 1
                time.sleep(2 ** attempt)  # 지수 백오프
                continue

            response.raise_for_status()
            logger.debug(f"API 호출 성공: {url}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.exception(f"API 호출 중 예외 발생: {e}. 재시도 시도 {attempt + 1}/{max_retries}")
            attempt += 1
            time.sleep(2 ** attempt)  # 지수 백오프
    logger.error(f"API 호출 실패: {url} - {method}")
    return None

def get_position(symbol, category="linear"):
    """포지션 조회"""
    logger.debug(f"get_position 호출 - symbol: {symbol}, category: {category}")
    endpoint = "/v5/position/list"
    params = {
        "apiKey": API_KEY,             # CamelCase로 수정
        "symbol": symbol,
        "category": category,          # 필수 파라미터
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000             # CamelCase로 수정
    }
    params["sign"] = generate_signature(params, API_SECRET)
    response = call_bybit_api(endpoint, method='GET', params=params)
    logger.debug(f"get_position 응답: {response}")
    return response

def get_wallet_balance(coin="USDT", account_type="CONTRACT"):
    """잔고 조회"""
    logger.debug(f"get_wallet_balance 호출 - coin: {coin}, account_type: {account_type}")
    endpoint = "/v5/account/wallet-balance"
    params = {
        "apiKey": API_KEY,             # CamelCase로 수정
        "coin": coin,
        "accountType": account_type,    # CamelCase로 수정
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000             # CamelCase로 수정
    }
    params["sign"] = generate_signature(params, API_SECRET)
    response = call_bybit_api(endpoint, method='GET', params=params)
    logger.debug(f"get_wallet_balance 응답: {response}")
    return response

def place_order(symbol, side, order_type, qty, leverage=5, reduce_only=False, category="linear"):
    """주문 생성"""
    logger.debug(f"place_order 호출 - symbol: {symbol}, side: {side}, order_type: {order_type}, qty: {qty}, leverage: {leverage}, reduce_only: {reduce_only}, category: {category}")
    endpoint = "/v5/order/create"
    params = {
        "apiKey": API_KEY,             # CamelCase로 수정
        "symbol": symbol,
        "side": side,                   # "Buy" or "Sell"
        "orderType": order_type,        # "Market"
        "qty": qty,
        "timeInForce": "GoodTillCancel",
        "reduceOnly": reduce_only,
        "category": category,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000              # CamelCase로 수정
    }
    # 레버리지는 새로운 포지션을 열 때만 포함
    if not reduce_only:
        params["leverage"] = leverage    # 레버리지 포함
    params["sign"] = generate_signature(params, API_SECRET)
    response = call_bybit_api(endpoint, method='POST', data=params)  # params 제거, data에만 전달
    logger.debug(f"place_order 응답: {response}")
    return response

def set_leverage(symbol, leverage=5, category="linear"):
    """레버리지 설정"""
    logger.debug(f"set_leverage 호출 - symbol: {symbol}, leverage: {leverage}, category: {category}")
    endpoint = "/v5/position/set-leverage"  # 올바른 엔드포인트 사용
    params = {
        "apiKey": API_KEY,             # CamelCase로 수정
        "symbol": symbol,
        "buyLeverage": leverage,
        "sellLeverage": leverage,
        "category": category,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000              # CamelCase로 수정
    }
    params["sign"] = generate_signature(params, API_SECRET)
    response = call_bybit_api(endpoint, method='POST', data=params)  # params 제거, data에만 전달
    logger.debug(f"set_leverage 응답: {response}")
    return response

# MongoDB 연결 설정
def init_db():
    logger.info("init_db 함수 시작")
    # 환경 변수에서 MongoDB 비밀번호 가져오기
    db_password = os.getenv("MONGODB_PASSWORD")
    if not db_password:
        logger.error("MongoDB password not found. Please set the MONGODB_PASSWORD environment variable.")
        raise ValueError("Missing MongoDB password.")

    logger.debug("MongoDB 비밀번호 로드 완료")
    
    # 비밀번호를 URL 인코딩
    encoded_password = quote_plus(db_password)

    # MongoDB 연결 URI 구성 (새 클러스터 주소와 데이터베이스 이름 반영)
    mongo_uri = f"mongodb+srv://juh4212:{encoded_password}@cluster0.7grdy.mongodb.net/bitcoin_trades_db?retryWrites=true&w=majority&appName=Cluster0&authSource=admin"

    logger.debug(f"MongoDB URI: {mongo_uri}")

    try:
        # MongoClient 생성 시 ServerApi 사용
        client = MongoClient(mongo_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        
        # 서버 정보 조회로 연결 확인
        client.admin.command('ping')
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        logger.info("MongoDB에 성공적으로 연결되었습니다.")
        return trades_collection
    except Exception as e:
        logger.exception(f"MongoDB 연결 실패: {e}")
        raise

# 거래 기록을 DB에 저장하는 함수
def log_trade(trades_collection, symbol, decision, percentage, reason, btc_balance,
              usdt_balance, btc_avg_buy_price, btc_usdt_price, reflection=''):
    logger.info("log_trade 함수 시작")
    trade = {
        "timestamp": datetime.now(),
        "symbol": symbol,  # 심볼 추가
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
        logger.info(f"{symbol} 거래 기록이 성공적으로 DB에 저장되었습니다.")
    except Exception as e:
        logger.exception(f"{symbol} 거래 기록 DB 저장 실패: {e}")

# 최근 투자 기록 조회
def get_recent_trades(trades_collection, symbol, days=7, limit=50):
    logger.info(f"get_recent_trades 함수 시작 - {symbol}의 최근 {days}일간의 거래 내역 조회")
    seven_days_ago = datetime.now() - timedelta(days=days)
    try:
        cursor = trades_collection.find({"symbol": symbol, "timestamp": {"$gte": seven_days_ago}}).sort("timestamp", -1).limit(limit)
        trades = list(cursor)
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            logger.info(f"{symbol}의 최근 거래 내역 조회 성공 - 총 {len(trades_df)}건")
        else:
            logger.info(f"{symbol}의 최근 거래 내역이 없습니다.")
        return trades_df
    except Exception as e:
        logger.exception(f"{symbol}의 최근 거래 내역 조회 실패: {e}")
        return pd.DataFrame()

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    logger.info("calculate_performance 함수 시작")
    if trades_df.empty:
        logger.info("거래 기록이 없어 퍼포먼스를 0%로 설정합니다.")
        return 0  # 기록이 없을 경우 0%로 설정
    try:
        # 초기 잔고 계산 (USDT + BTC * 당시 가격)
        initial_balance = trades_df.iloc[-1]['usdt_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_usdt_price']
        # 최종 잔고 계산
        final_balance = trades_df.iloc[0]['usdt_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_usdt_price']
        performance = (final_balance - initial_balance) / initial_balance * 100
        logger.info(f"퍼포먼스 계산 완료: {performance:.2f}%")
        return performance
    except Exception as e:
        logger.exception(f"퍼포먼스 계산 실패: {e}")
        return 0

# 퍼포먼스 기반 포지션 크기 조정 함수
def adjust_position_size(performance, base_percentage=20):
    """
    퍼포먼스에 따라 포지션 크기 조정
    - 손실 시 포지션 크기 10% 감소
    - 이익 시 포지션 크기 10% 증가
    """
    logger.debug(f"adjust_position_size 호출 - performance: {performance}, base_percentage: {base_percentage}")
    if performance < 0:
        # 손실이 발생했을 경우 포지션 크기 10% 감소
        adjusted_percentage = max(10, base_percentage - 10)
        logger.info(f"퍼포먼스가 음수이므로 진입 비율을 {adjusted_percentage}%로 감소시킵니다.")
        return adjusted_percentage
    elif performance > 0:
        # 이익이 발생했을 경우 포지션 크기 10% 증가
        adjusted_percentage = min(30, base_percentage + 10)
        logger.info(f"퍼포먼스가 양수이므로 진입 비율을 {adjusted_percentage}%로 증가시킵니다.")
        return adjusted_percentage
    logger.info(f"퍼포먼스 변화 없음. 진입 비율을 {base_percentage}%로 유지합니다.")
    return base_percentage

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(symbol, trades_df, current_market_data):
    logger.info(f"generate_reflection 함수 시작 - {symbol}")
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key가 누락되었거나 유효하지 않습니다.")
        return None

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    try:
        # 프롬프트 최적화를 위해 예시 응답 간소화
        examples = """
decision: open_long
percentage: 20
leverage: 5
reason: Market indicators are favorable, entering a long position with moderate leverage.
        """

        # 현재 포지션 상태 확인
        current_position = {
            "long": bool(trades_df['btc_balance'].iloc[-1] > 0) if not trades_df.empty else False,
            "short": bool(trades_df['btc_balance'].iloc[-1] < 0) if not trades_df.empty else False
        }
        logger.debug(f"현재 포지션 상태: {current_position}")

        # AI 프롬프트 수정: 'hold' 옵션 제거 (테스트 모드 기준)
        if TEST_MODE:
            # 테스트 모드에서는 'hold' 제외
            if current_position["long"]:
                possible_decisions = """
Possible decisions:
- close_long
- open_short
"""
            elif current_position["short"]:
                possible_decisions = """
Possible decisions:
- close_short
- open_long
"""
            else:
                possible_decisions = """
Possible decisions:
- open_long
- open_short
"""
        else:
            # 운영 모드에서는 기존 'hold' 옵션 포함
            if current_position["long"]:
                possible_decisions = """
Possible decisions:
- close_long
- hold
"""
            elif current_position["short"]:
                possible_decisions = """
Possible decisions:
- close_short
- hold
"""
            else:
                possible_decisions = """
Possible decisions:
- open_long
- open_short
- hold
"""

        # 퍼포먼스 기반 포지션 크기 조정
        adjusted_percentage = adjust_position_size(performance)
        logger.info(f"조정된 진입 비율: {adjusted_percentage}%")

        # 데이터 축소: 오더북 상위 5개 호가, 핵심 지표만 포함
        reduced_orderbook = {
            "bids": current_market_data['orderbook']['bids'][:5],
            "asks": current_market_data['orderbook']['asks'][:5]
        }

        # 필요한 지표만 포함 (예: RSI)
        rsi_daily = current_market_data['daily_ohlcv'].get('rsi', {}).get('mean', 0)
        rsi_hourly = current_market_data['hourly_ohlcv'].get('rsi', {}).get('mean', 0)
        rsi_4h = current_market_data['four_hour_ohlcv'].get('rsi', {}).get('mean', 0)

        # fib_high_4h와 fib_low_4h를 현재 시장 데이터에 추가
        fib_high_4h = current_market_data['four_hour_ohlcv'].get('fib_high_4h', 0)
        fib_low_4h = current_market_data['four_hour_ohlcv'].get('fib_low_4h', 0)

        # AI 프롬프트 최적화: 필요한 정보만 포함
        prompt = f"""
{possible_decisions}

레버리지는 5배로 고정하며, 새로운 포지션을 열 때만 포함합니다.
진입 비율은 {adjusted_percentage}%로 설정합니다.
수수료는 0.055%로 계산하며, 레버리지를 곱해서 적용합니다.

시장 신호가 명확하지 않거나 애매한 경우에도 'open_long' 또는 'open_short' 결정을 내려주세요.

응답 형식 (예시 참고):
{examples}

---

포지션: 롱 - {current_position['long']}, 숏 - {current_position['short']}
USDT 잔고: {current_market_data['usdt_balance']}

오더북 상위 5개 호가:
Bids: {', '.join([f"{bid[0]}@{bid[1]}" for bid in reduced_orderbook['bids']])}
Asks: {', '.join([f"{ask[0]}@{ask[1]}" for ask in reduced_orderbook['asks']])}

일일 RSI: {rsi_daily}
시간별 RSI: {rsi_hourly}
4시간 RSI: {rsi_4h}

4시간봉 피보나치 고가 EMA 평균: {fib_high_4h}
4시간봉 피보나치 저가 EMA 평균: {fib_low_4h}

이전 퍼포먼스: {performance:.2f}%
"""

        logger.debug(f"AI 요청 프롬프트:\n{prompt}")

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 암호화폐 선물 트레이딩 전문가입니다. 제공된 데이터를 분석하고 현재 시점에서 최선의 행동을 결정하세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,  # 토큰 수 줄이기
            n=1,
            stop=None,
            temperature=0.2  # 응답의 창의성 조절
        )
        logger.info("OpenAI API 호출 성공")
        response_content = response.choices[0].message.content
        logger.debug(f"AI 응답 전체 내용: {response_content}")
        return response_content
    except Exception as e:
        logger.exception(f"OpenAI API 호출 실패: {e}")
        return None

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df, higher_timeframe_df, timeframe="60"):
    logger.info("add_indicators 함수 시작")
    try:
        # 기존 지표들
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

        # 새로운 지표 추가
        # 1. Ichimoku Cloud (일목균형표)
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['high'],
            low=df['low'],
            window1=9,
            window2=26,
            window3=52
        )
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()

        # 2. VWAP (Volume Weighted Average Price)
        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=14
        )
        df['vwap'] = vwap.volume_weighted_average_price()

        # 3. Chaikin Money Flow (CMF)
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=20
        )
        df['cmf'] = cmf.chaikin_money_flow()

        # 4. 피보나치 EMA 추가 (4시간 차트)
        # 피보나치 수열 기반 EMA 기간 리스트
        fib_periods = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
        fib_ema_prefix = "fib_ema_"
        for period in fib_periods:
            ema_col_high = fib_ema_prefix + f"high_{period}"
            ema_col_low = fib_ema_prefix + f"low_{period}"
            # 4시간봉 고가 EMA 계산
            df[ema_col_high] = ta.trend.EMAIndicator(close=higher_timeframe_df['high'], window=period).ema_indicator()
            # 4시간봉 저가 EMA 계산
            df[ema_col_low] = ta.trend.EMAIndicator(close=higher_timeframe_df['low'], window=period).ema_indicator()

        # 4시간봉 고가 EMA 평균 계산
        high_ema_cols = [fib_ema_prefix + f"high_{period}" for period in fib_periods]
        df['fib_high_4h'] = df[high_ema_cols].mean(axis=1)

        # 4시간봉 저가 EMA 평균 계산
        low_ema_cols = [fib_ema_prefix + f"low_{period}" for period in fib_periods]
        df['fib_low_4h'] = df[low_ema_cols].mean(axis=1)

        logger.info("보조 지표 추가 완료")
        return df
    except Exception as e:
        logger.exception(f"보조 지표 추가 실패: {e}")
        return df

# 가격 데이터 가져오기 함수 (Bybit V5 API 사용)
def get_ohlcv(symbol, interval, limit, category="linear"):
    logger.info(f"get_ohlcv 함수 시작 - symbol: {symbol}, interval: {interval}, limit: {limit}")
    endpoint = "/v5/market/kline"
    params = {
        "category": category,  # 필수 파라미터
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = call_bybit_api(endpoint, method='GET', params=params)
    logger.debug(f"get_ohlcv 응답: {response}")
    if response and response.get('retCode') == 0:
        try:
            records = response['result']['list']
            df = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            # 밀리초 단위를 지정하여 datetime 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            logger.info(f"OHLCV 데이터 조회 성공 - {symbol} - 총 {len(df)}건")
            return df
        except Exception as e:
            logger.exception(f"OHLCV 데이터 처리 실패: {e}")
            return None
    else:
        logger.error(f"OHLCV 데이터 조회 오류: {response.get('retMsg') if response else 'No response'}")
        return None

# AI 결정 실행 전 검증 함수
def validate_decision(decision, current_position):
    """
    AI의 결정이 현재 포지션과 충돌하지 않는지 검증합니다.
    """
    logger.debug(f"validate_decision 호출 - decision: {decision}, current_position: {current_position}")
    if decision == "open_long" and current_position["short"]:
        logger.warning("이미 숏 포지션에 있으므로 롱 포지션을 열 수 없습니다.")
        return False
    if decision == "open_short" and current_position["long"]:
        logger.warning("이미 롱 포지션에 있으므로 숏 포지션을 열 수 없습니다.")
        return False
    if decision in ["close_long", "close_short"] and not (current_position["long"] or current_position["short"]):
        logger.warning("청산할 포지션이 없으므로 해당 결정을 실행할 수 없습니다.")
        return False
    logger.debug("결정이 유효함")
    return True

### 메인 AI 트레이딩 로직
def ai_trading():
    logger.info("ai_trading 함수 시작")
    trades_collection = None
    try:
        # 데이터베이스 연결
        logger.info("데이터베이스 연결 시도")
        trades_collection = init_db()
    except Exception as e:
        logger.exception(f"데이터베이스 연결 실패: {e}")
        return

    symbol = "BTCUSDT"  # 비트코인 심볼로 고정

    logger.info(f"{symbol}에 대한 트레이딩 시작")
    ### 데이터 가져오기
    # 1. 현재 포지션 조회 (Bybit V5 API 사용)
    try:
        logger.info(f"{symbol} 현재 포지션 조회 시도")
        response = get_position(symbol, category="linear")
        logger.debug(f"{symbol} 포지션 조회 응답: {response}")
        if not response or response.get('retCode') != 0:
            logger.error(f"{symbol} 포지션 조회 오류: {response.get('retMsg') if response else 'No response'}")
            return
        positions = response['result']['list']
        # 포지션 정보 파싱
        long_position = next((p for p in positions if p['side'] == 'Buy'), None)
        short_position = next((p for p in positions if p['side'] == 'Sell'), None)
        logger.debug(f"{symbol} 롱 포지션: {long_position}, 숏 포지션: {short_position}")
    except Exception as e:
        logger.exception(f"{symbol} 포지션 조회 실패: {e}")
        return

    # 2. 현재 잔고 조회 (Bybit V5 API 사용)
    try:
        logger.info(f"{symbol} 현재 잔고 조회 시도")
        response = get_wallet_balance("USDT", account_type="CONTRACT")
        logger.debug(f"{symbol} 잔고 조회 응답: {response}")
        if not response or response.get('retCode') != 0:
            logger.error(f"{symbol} 잔고 조회 오류: {response.get('retMsg') if response else 'No response'}")
            return

        usdt_balance = None
        for item in response.get('result', {}).get('list', []):
            coin_info = item.get('coin', [])
            if isinstance(coin_info, list):
                for coin in coin_info:
                    if isinstance(coin, dict) and coin.get('coin') == 'USDT':
                        usdt_balance = float(coin.get('availableToWithdraw', '0'))
                        break
            elif isinstance(coin_info, dict):
                if coin_info.get('coin') == 'USDT':
                    usdt_balance = float(coin_info.get('availableToWithdraw', '0'))
            if usdt_balance is not None:
                break

        if usdt_balance is not None:
            logger.info(f"{symbol} USDT 잔고: {usdt_balance}")
        else:
            logger.error(f"{symbol} USDT 잔고 정보를 찾을 수 없습니다.")
            return
    except Exception as e:
        logger.exception(f"{symbol} 잔고 조회 실패: {e}")
        return

    # 3. 오더북(호가 데이터) 조회 (Bybit V5 API 사용)
    try:
        logger.info(f"{symbol} 오더북 조회 시도")
        response = call_bybit_api("/v5/market/orderbook", method='GET', params={"symbol": symbol, "limit": 10, "category": "linear"})
        logger.debug(f"{symbol} 오더북 조회 응답: {response}")
        if not response or response.get('retCode') != 0:
            logger.error(f"{symbol} 오더북 조회 오류: {response.get('retMsg') if response else 'No response'}")
            orderbook = {}
        else:
            orderbook = {
                'bids': response['result']['b'],
                'asks': response['result']['a']
            }
            logger.debug(f"{symbol} 오더북 데이터: {orderbook}")
    except Exception as e:
        logger.exception(f"{symbol} 오더북 조회 실패: {e}")
        orderbook = {}

    # 4. 차트 데이터 조회 및 보조지표 추가 (Bybit V5 API 사용)
    try:
        logger.info(f"{symbol} 차트 데이터 조회 시도 - 일일 데이터")
        df_daily = get_ohlcv(symbol, interval="D", limit=60, category="linear")  # 데이터 양 축소
        if df_daily is None:
            logger.error(f"{symbol} 일일 OHLCV 데이터 조회 실패")
            return
        df_daily = dropna(df_daily)
        df_daily = add_indicators(df_daily, df_daily, timeframe="D")  # 일일 데이터에 지표 추가

        logger.info(f"{symbol} 차트 데이터 조회 시도 - 시간별 데이터")
        df_hourly = get_ohlcv(symbol, interval="60", limit=48, category="linear")  # 2일치 데이터로 축소
        if df_hourly is None:
            logger.error(f"{symbol} 시간별 OHLCV 데이터 조회 실패")
            return
        df_hourly = dropna(df_hourly)
        df_hourly = add_indicators(df_hourly, df_hourly, timeframe="60")  # 시간별 데이터에 지표 추가

        # 4시간 데이터 추가
        logger.info(f"{symbol} 차트 데이터 조회 시도 - 4시간 데이터")
        df_4h = get_ohlcv(symbol, interval="240", limit=50, category="linear")  # 데이터 양 축소
        if df_4h is None:
            logger.error(f"{symbol} 4시간 OHLCV 데이터 조회 실패")
            return
        df_4h = dropna(df_4h)
        df_4h = add_indicators(df_4h, df_4h, timeframe="240")  # 4시간 데이터에 지표 추가

        # 최근 데이터만 사용하도록 설정 (메모리 절약)
        df_daily_recent = df_daily.tail(60)
        df_hourly_recent = df_hourly.tail(48)
        df_4h_recent = df_4h.tail(50)  # 피보나치 EMA를 계산하기 위해 데이터 양 축소
        logger.info(f"{symbol} 최근 일일 데이터: {df_daily_recent.shape[0]}건, 최근 시간별 데이터: {df_hourly_recent.shape[0]}건, 4시간 데이터: {df_4h_recent.shape[0]}건")
    except Exception as e:
        logger.exception(f"{symbol} 차트 데이터 조회 또는 보조지표 추가 실패: {e}")
        return

    ### AI에게 데이터 제공하고 판단 받기
    try:
        # 최근 거래 내역 가져오기
        logger.info(f"{symbol} 최근 거래 내역 조회 시도")
        recent_trades = get_recent_trades(trades_collection, symbol)

        # 현재 시장 데이터 수집
        current_market_data = {
            "usdt_balance": usdt_balance,
            "orderbook": orderbook,
            "daily_ohlcv": df_daily_recent.describe().to_dict(),  # 요약 통계로 대체
            "hourly_ohlcv": df_hourly_recent.describe().to_dict(),  # 요약 통계로 대체
            "four_hour_ohlcv": {
                "rsi": df_4h_recent['rsi'].mean(),
                "fib_high_4h": df_4h_recent['fib_high_4h'].iloc[-1],
                "fib_low_4h": df_4h_recent['fib_low_4h'].iloc[-1]
            }  # 피보나치 EMA 추가를 위해 4H 데이터 요약
        }
        logger.debug(f"{symbol} 현재 시장 데이터: {current_market_data}")

        # 반성 및 개선 내용 생성
        logger.info(f"{symbol} 반성 및 개선 내용 생성 시도")
        reflection = generate_reflection(symbol, recent_trades, current_market_data)
        logger.debug(f"{symbol} 생성된 반성 내용: {reflection}")

        if not reflection:
            logger.error(f"{symbol} 반성 내용을 생성할 수 없습니다.")
            return

        # AI 모델에 반성 내용 제공 및 투자 판단 받기
        logger.info(f"{symbol} OpenAI API 호출 준비 완료")

        response_text = reflection

        # AI 응답 파싱
        def parse_ai_response(response_text):
            logger.info("parse_ai_response 함수 시작")
            try:
                # JSON 부분 추출 시도
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    decision = parsed_json.get('decision')
                    percentage = parsed_json.get('percentage')
                    leverage = parsed_json.get('leverage')
                    reason = parsed_json.get('reason')
                    logger.debug(f"파싱된 JSON: {parsed_json}")
                    # 모든 필드가 존재하는지 확인
                    if decision and percentage and leverage and reason:
                        return {'decision': decision, 'percentage': percentage, 'leverage': leverage, 'reason': reason}
                    else:
                        logger.warning("JSON 응답에 필요한 필드가 누락되었습니다.")
                else:
                    # JSON 형식이 아닐 경우 텍스트에서 정보 추출
                    logger.warning("응답이 JSON 형식이 아님. 텍스트에서 정보 추출 시도")
                    decision_match = re.search(r'decision:\s*(\w+)', response_text, re.IGNORECASE)
                    percentage_match = re.search(r'percentage:\s*(\d+)', response_text, re.IGNORECASE)
                    leverage_match = re.search(r'leverage:\s*(\d+)', response_text, re.IGNORECASE)
                    reason_match = re.search(r'reason:\s*(.+)', response_text, re.IGNORECASE)
                    decision = decision_match.group(1).lower() if decision_match else None
                    percentage = int(percentage_match.group(1)) if percentage_match else None
                    leverage = int(leverage_match.group(1)) if leverage_match else None
                    reason = reason_match.group(1).strip() if reason_match else None
                    parsed_data = {'decision': decision, 'percentage': percentage, 'leverage': leverage, 'reason': reason}
                    logger.debug(f"파싱된 텍스트 데이터: {parsed_data}")
                    # 모든 필드가 존재하는지 확인
                    if decision and percentage and leverage and reason:
                        return parsed_data
                    else:
                        logger.warning("텍스트 응답에 필요한 필드가 누락되었습니다.")
                # 필요한 필드가 모두 존재하지 않으면 None 반환
                return None
            except Exception as e:
                logger.exception(f"AI 응답 파싱 실패: {e}")
                return None

        parsed_response = parse_ai_response(response_text)
        if not parsed_response:
            logger.error(f"{symbol} AI 응답에 불완전한 데이터가 포함되어 있습니다. 기본적으로 'open_long' 결정을 내립니다.")
            decision = "open_long"
            percentage = 20
            leverage = 5  # 기본 레버리지 설정
            reason = "AI 응답이 불완전하여 자동으로 'open_long' 결정."
        else:
            decision = parsed_response.get('decision')
            percentage = parsed_response.get('percentage')
            leverage = parsed_response.get('leverage')
            reason = parsed_response.get('reason')

            if not decision or reason is None or percentage is None:
                logger.error(f"{symbol} AI 응답에 불완전한 데이터가 포함되어 있습니다. 기본적으로 'open_long' 결정을 내립니다.")
                decision = "open_long"
                percentage = 20
                leverage = 5
                reason = "AI 응답이 불완전하여 자동으로 'open_long' 결정."

        logger.info(f"{symbol} AI Decision: {decision.upper()}")
        logger.info(f"{symbol} Percentage: {percentage}%")
        if leverage:
            logger.info(f"{symbol} Leverage: {leverage}x")
        logger.info(f"{symbol} Decision Reason: {reason}")

        # 신호의 명확성 평가 (예시: 특정 지표의 기준 미달 시 "open_long" 또는 "open_short"로 변경)
        # 이 부분은 실제 전략에 맞게 구현해야 합니다.
        # 예를 들어, RSI가 과매수/과매도 범위에 있지 않을 경우 신호를 변경
        rsi = current_market_data['daily_ohlcv'].get('rsi', {}).get('mean', 50)
        logger.debug(f"일일 RSI: {rsi}")
        if not (30 < rsi < 70):
            logger.info(f"RSI가 {rsi}로, 신호가 애매하여 'open_long' 또는 'open_short'로 결정합니다.")
            # 임의로 'open_long' 또는 'open_short'를 선택하도록 수정
            decision = "open_long" if rsi > 50 else "open_short"

        # 현재 포지션 상태
        current_position = {
            "long": bool(long_position and float(long_position['size']) > 0),
            "short": bool(short_position and float(short_position['size']) > 0)
        }
        logger.debug(f"현재 포지션 상태 (검증 전): {current_position}")

        # AI 결정 검증 및 강제 변경
        if not validate_decision(decision, current_position):
            logger.warning(f"{symbol} 유효하지 않은 결정이므로 기본 결정으로 변경합니다.")
            # 기본 결정을 'open_long' 또는 'open_short'로 설정
            decision = "open_long" if rsi > 50 else "open_short"

        order_executed = False

        # 현재 가격 가져오기 (Bybit V5 API 사용)
        try:
            logger.info(f"{symbol} 현재 가격 데이터 조회 시도")
            response = call_bybit_api("/v5/market/tickers", method='GET', params={"symbol": symbol, "category": "linear"})
            logger.debug(f"{symbol} 현재 가격 조회 응답: {response}")
            if not response or response.get('retCode') != 0:
                logger.error(f"{symbol} 현재 가격 조회 오류: {response.get('retMsg') if response else 'No response'}")
                return
            current_price = float(response['result']['list'][0]['lastPrice'])
            logger.info(f"{symbol} 현재 가격: {current_price}")
        except Exception as e:
            logger.exception(f"{symbol} 현재 가격 데이터 조회 실패: {e}")
            return

        # 주문 실행 (Bybit V5 API 사용)
        try:
            if decision == "open_long":
                # 레버리지 확인
                if leverage is None:
                    logger.error(f"{symbol} 레버리지가 필요합니다. 포지션을 여는 데 실패했습니다.")
                    return
                # 레버리지를 5으로 고정
                leverage = 5
                logger.info(f"{symbol} 설정된 레버리지: {leverage}x")

                try:
                    # 레버리지 설정
                    response = set_leverage(symbol=symbol, leverage=leverage, category="linear")
                    if not response or response.get('retCode') != 0:
                        logger.error(f"{symbol} 레버리지 설정 오류: {response.get('retMsg') if response else 'No response'}")
                        return
                    logger.info(f"{symbol} 레버리지 설정 완료: {leverage}x")
                except Exception as e:
                    logger.exception(f"{symbol} 레버리지 설정 실패: {e}")
                    return

                # 진입 비율을 10%에서 30% 사이로 제한
                percentage = max(10, min(int(percentage), 30))
                logger.info(f"{symbol} 설정된 진입 비율: {percentage}%")

                # 포지션 크기 계산: 10%에서 30% 사이의 USDT 잔고
                position_size = usdt_balance * (percentage / 100) * 0.9995  # 수수료 고려
                # 수수료 계산: 0.055% * 레버리지
                fee = position_size * 0.00055 * leverage
                position_size_after_fee = position_size - fee
                logger.info(f"{symbol} 포지션 크기 (수수료 포함 후): {position_size_after_fee} USDT, 수수료: {fee} USDT")

                if position_size_after_fee > 10:  # 최소 거래 금액은 거래소에 따라 다를 수 있음
                    logger.info(f"{symbol} 롱 포지션 주문 시도: {percentage}%의 USDT와 {leverage}x 레버리지")
                    order_qty = round((position_size_after_fee * leverage) / current_price, 6)  # 레버리지 적용
                    logger.debug(f"{symbol} 주문 수량 계산: {order_qty}")
                    try:
                        order = place_order(
                            symbol=symbol,
                            side="Buy",
                            order_type="Market",
                            qty=order_qty,
                            leverage=leverage,
                            reduce_only=False,
                            category="linear"
                        )
                        if order and order.get('retCode') == 0:
                            logger.info(f"{symbol} 롱 포지션 주문 성공: {order}")
                            order_executed = True
                        else:
                            logger.error(f"{symbol} 롱 포지션 주문 실패: {order.get('retMsg') if order else 'No response'}")
                    except Exception as e:
                        logger.exception(f"{symbol} 롱 포지션 주문 중 오류 발생: {e}")
                else:
                    logger.warning(f"{symbol} 롱 주문 실패: USDT 잔고가 부족합니다.")
            elif decision == "open_short":
                # 레버리지 확인
                if leverage is None:
                    logger.error(f"{symbol} 레버리지가 필요합니다. 포지션을 여는 데 실패했습니다.")
                    return
                # 레버리지를 5으로 고정
                leverage = 5
                logger.info(f"{symbol} 설정된 레버리지: {leverage}x")

                try:
                    # 레버리지 설정
                    response = set_leverage(symbol=symbol, leverage=leverage, category="linear")
                    if not response or response.get('retCode') != 0:
                        logger.error(f"{symbol} 레버리지 설정 오류: {response.get('retMsg') if response else 'No response'}")
                        return
                    logger.info(f"{symbol} 레버리지 설정 완료: {leverage}x")
                except Exception as e:
                    logger.exception(f"{symbol} 레버리지 설정 실패: {e}")
                    return

                # 진입 비율을 10%에서 30% 사이로 제한
                percentage = max(10, min(int(percentage), 30))
                logger.info(f"{symbol} 설정된 진입 비율: {percentage}%")

                # 포지션 크기 계산: 10%에서 30% 사이의 USDT 잔고
                position_size = usdt_balance * (percentage / 100) * 0.9995  # 수수료 고려
                # 수수료 계산: 0.055% * 레버리지
                fee = position_size * 0.00055 * leverage
                position_size_after_fee = position_size - fee
                logger.info(f"{symbol} 포지션 크기 (수수료 포함 후): {position_size_after_fee} USDT, 수수료: {fee} USDT")

                if position_size_after_fee > 10:
                    logger.info(f"{symbol} 숏 포지션 주문 시도: {percentage}%의 USDT와 {leverage}x 레버리지")
                    order_qty = round((position_size_after_fee * leverage) / current_price, 6)
                    logger.debug(f"{symbol} 주문 수량 계산: {order_qty}")
                    try:
                        order = place_order(
                            symbol=symbol,
                            side="Sell",
                            order_type="Market",
                            qty=order_qty,
                            leverage=leverage,
                            reduce_only=False,
                            category="linear"
                        )
                        if order and order.get('retCode') == 0:
                            logger.info(f"{symbol} 숏 포지션 주문 성공: {order}")
                            order_executed = True
                        else:
                            logger.error(f"{symbol} 숏 포지션 주문 실패: {order.get('retMsg') if order else 'No response'}")
                    except Exception as e:
                        logger.exception(f"{symbol} 숏 포지션 주문 중 오류 발생: {e}")
                else:
                    logger.warning(f"{symbol} 숏 주문 실패: USDT 잔고가 부족합니다.")
            else:
                logger.error(f"{symbol} AI로부터 유효하지 않은 결정을 받았습니다.")
                return

            # 거래 실행 여부와 관계없이 현재 잔고 및 포지션 조회
            logger.info(f"{symbol} 거래 후 잔고 및 포지션 조회 시도")
            time.sleep(2)  # API 호출 제한을 고려하여 잠시 대기
            try:
                # 포지션 재조회
                response = get_position(symbol, category="linear")
                logger.debug(f"{symbol} 포지션 재조회 응답: {response}")
                if not response or response.get('retCode') != 0:
                    logger.error(f"{symbol} 포지션 재조회 오류: {response.get('retMsg') if response else 'No response'}")
                    return
                positions = response['result']['list']
                long_position = next((p for p in positions if p['side'] == 'Buy'), None)
                short_position = next((p for p in positions if p['side'] == 'Sell'), None)

                # 잔고 재조회
                response = get_wallet_balance("USDT", account_type="CONTRACT")
                logger.debug(f"{symbol} 잔고 재조회 응답: {response}")
                if not response or response.get('retCode') != 0:
                    logger.error(f"{symbol} 잔고 재조회 오류: {response.get('retMsg') if response else 'No response'}")
                    return
                usdt_balance = None
                for item in response.get('result', {}).get('list', []):
                    coin_info = item.get('coin', [])
                    if isinstance(coin_info, list):
                        for coin in coin_info:
                            if isinstance(coin, dict) and coin.get('coin') == 'USDT':
                                usdt_balance = float(coin.get('availableToWithdraw', '0'))
                                break
                    elif isinstance(coin_info, dict):
                        if coin_info.get('coin') == 'USDT':
                            usdt_balance = float(coin_info.get('availableToWithdraw', '0'))
                    if usdt_balance is not None:
                        break

                if usdt_balance is not None:
                    logger.info(f"{symbol} USDT 잔고: {usdt_balance}")
                else:
                    logger.error(f"{symbol} USDT 잔고 정보를 찾을 수 없습니다.")
                    return

                # BTC 잔고 계산
                btc_balance = (float(long_position['size']) if long_position else 0) - \
                              (float(short_position['size']) if short_position else 0)
                btc_avg_buy_price = float(long_position['avgPrice']) if long_position else None
                current_btc_price = current_price

                # 거래 기록을 DB에 저장하기
                log_trade(
                    trades_collection,
                    symbol,
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
                logger.exception(f"{symbol} 거래 후 잔고 및 포지션 조회 실패: {e}")
        except Exception as e:
            logger.exception(f"{symbol} 주문 실행 중 오류 발생: {e}")
            return

    except Exception as e:
        logger.exception(f"{symbol} AI 트레이딩 로직 중 오류 발생: {e}")
        return

    logger.info(f"{symbol} ai_trading 함수 종료")

if __name__ == "__main__":
    try:
        logger.info("메인 스크립트 시작")
        # 환경 변수 확인
        logger.debug(f"BYBIT_API_KEY: {API_KEY}")
        logger.debug(f"BYBIT_API_SECRET: {'***' if API_SECRET else 'None'}")
        logger.debug(f"MONGODB_PASSWORD: {'***' if os.getenv('MONGODB_PASSWORD') else 'None'}")
        logger.debug(f"OPENAI_API_KEY: {'***' if os.getenv('OPENAI_API_KEY') else 'None'}")
        logger.debug(f"TEST_MODE: {TEST_MODE}")

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
                logger.info("트레이딩 작업 시작")
                ai_trading()
            except Exception as e:
                logger.exception(f"트레이딩 작업 중 오류 발생: {e}")
            finally:
                trading_in_progress = False
                logger.info("트레이딩 작업 종료")

        # 초기 실행 후 매 5분마다 실행되도록 스케줄링 설정
        def initial_and_recurring_schedule():
            # 첫 번째 실행: 1분 후
            schedule.every(1).minutes.do(first_run).tag('first_run')

        def first_run():
            job()
            # 이후 매 5분마다 실행되도록 설정
            schedule.every(5).minutes.do(job)
            # 첫 번째 실행 스케줄 제거
            schedule.clear('first_run')
            logger.info("초기 실행 완료. 이후부터는 매 5분마다 실행됩니다.")

        # 스케줄링 초기화
        initial_and_recurring_schedule()

        logger.info("스케줄러 설정 완료")

        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.exception(f"스케줄러 루프 중 오류 발생: {e}")
                logger.info("잠시 대기 후 재시작합니다.")
                time.sleep(5)  # 잠시 대기 후 재시작
    except Exception as e:
        logger.exception(f"메인 스크립트 중 오류 발생: {e}")
