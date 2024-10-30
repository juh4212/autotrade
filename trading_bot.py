import os
import logging
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from datetime import datetime
from pybit.unified_trading import HTTP  # Bybit v5 API 사용
import threading
import pandas as pd
import ta  # 기술 지표 계산을 위한 라이브러리
import openai  # OpenAI API 사용
import re  # 정규 표현식 사용
import json  # JSON 파싱을 위한 라이브러리

# 환경 변수 로드
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB 설정 및 연결
def setup_mongodb():
    mongo_uri = os.getenv("MONGODB_URI")
    print("MongoDB URI:", mongo_uri)  # URI 확인을 위해 출력
    try:
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        # 필요한 인덱스 생성 (예: timestamp 인덱스)
        trades_collection.create_index([('timestamp', ASCENDING)])
        logger.info("MongoDB 연결 및 초기화 완료!")
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

# OpenAI API 설정
def setup_openai():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.critical("OpenAI API 키가 설정되지 않았습니다.")
        raise ValueError("OpenAI API 키가 누락되었습니다.")
    openai.api_key = openai_api_key
    logger.info("OpenAI API 설정 완료!")

# 도우미 함수 정의

def get_current_timestamp():
    """
    현재 UTC 시간을 ISO 형식으로 반환합니다.
    """
    return datetime.utcnow().isoformat()

def validate_balance_data(balance_data):
    """
    잔고 데이터의 유효성을 검증합니다.
    """
    if not balance_data:
        logger.error("잔고 데이터가 비어 있습니다.")
        return False
    # 추가적인 검증 로직을 여기에 추가할 수 있습니다.
    required_keys = ["equity", "available_to_withdraw"]
    for key in required_keys:
        if key not in balance_data:
            logger.error(f"잔고 데이터에 '{key}' 키가 없습니다.")
            return False
    return True

def handle_error(e, context=""):
    """
    공통 에러 핸들링 함수.
    """
    if context:
        logger.error(f"{context} 오류: {e}")
    else:
        logger.error(f"오류 발생: {e}")

def log_event(message, level="info"):
    """
    특정 이벤트를 로그로 기록하는 함수.
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.debug(message)

# MongoDB에 잔고 기록
def log_balance_to_mongodb(collection, balance_data):
    if not validate_balance_data(balance_data):
        return
    balance_record = {
        "timestamp": get_current_timestamp(),
        "balance_data": balance_data
    }
    try:
        collection.insert_one(balance_record)
        logger.info("계좌 잔고가 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        handle_error(e, "MongoDB에 계좌 잔고 저장")

# 현재 포지션 조회 함수 추가
def get_current_position(bybit, symbol="BTCUSDT"):
    """
    현재 포지션을 조회하는 함수.
    
    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")
        
    Returns:
        str: 현재 포지션 ('long', 'short', 'none') 또는 None
    """
    try:
        response = bybit.my_position(symbol=symbol)
        logger.debug(f"get_current_position API 응답: {response}")
        
        if response['retCode'] != 0:
            logger.error(f"포지션 조회 실패: {response['retMsg']}")
            return None
        
        positions = response.get('result', [])
        if not positions:
            logger.info("현재 포지션이 없습니다.")
            return "none"
        
        position = positions[0]  # 첫 번째 포지션을 기준으로 함
        side = position.get('side')
        if side == "Buy":
            return "long"
        elif side == "Sell":
            return "short"
        else:
            return "none"
    except Exception as e:
        handle_error(e, "get_current_position 함수")
        return None

# 글로벌 변수 및 락 설정
trading_in_progress = False
trading_lock = threading.Lock()

def job(bybit, collection):
    """
    스케줄링된 트레이딩 작업을 수행하는 함수.
    """
    global trading_in_progress
    with trading_lock:
        if trading_in_progress:
            logger.warning("이미 트레이딩 작업이 진행 중입니다. 현재 작업을 건너뜁니다.")
            return
        trading_in_progress = True

    try:
        logger.info("트레이딩 작업 시작...")
        ai_trading(bybit, collection)
        logger.info("트레이딩 작업 완료.")
    except Exception as e:
        handle_error(e, "job 함수")
    finally:
        with trading_lock:
            trading_in_progress = False

# ai_trading 함수 정의
def ai_trading(bybit, collection):
    """
    AI 기반 트레이딩 로직을 수행하는 함수.
    """
    try:
        # 1. 현재 잔고 조회
        balance_data = get_account_balance(bybit)
        if not balance_data:
            logger.error("잔고 데이터를 가져오지 못했습니다.")
            return

        # 2. 오더북 데이터 조회
        order_book = get_order_book(bybit)
        if not order_book:
            logger.error("오더북 데이터를 가져오지 못했습니다.")
            return

        # 3. 일별 OHLCV 데이터 조회 및 기술 지표 추가
        daily_ohlcv = get_daily_ohlcv(bybit, symbol="BTCUSDT", interval="D", limit=100)
        if daily_ohlcv is None:
            logger.error("일별 OHLCV 데이터를 가져오지 못했습니다.")
            return

        # 4. 시간별 OHLCV 데이터 조회 및 기술 지표 추가
        hourly_ohlcv = get_hourly_ohlcv(bybit, symbol="BTCUSDT", interval="60", limit=100)
        if hourly_ohlcv is None:
            logger.error("시간별 OHLCV 데이터를 가져오지 못했습니다.")
            return

        # 5. 현재 포지션 조회
        current_position = get_current_position(bybit, symbol="BTCUSDT")
        if current_position is None:
            logger.error("현재 포지션을 조회하지 못했습니다.")
            return

        # 6. AI를 사용하여 트레이딩 결정 요청
        trading_decision = request_ai_trading_decision(
            collection,
            balance_data,
            daily_ohlcv,
            hourly_ohlcv
        )
        if not trading_decision:
            logger.error("AI로부터 트레이딩 결정을 받지 못했습니다.")
            return

        # 7. AI 응답 파싱 및 결정 실행
        execute_trading_decision(bybit, collection, trading_decision, balance_data, current_position)

    except Exception as e:
        handle_error(e, "ai_trading 함수")

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

def get_order_book(bybit, symbol="BTCUSDT"):
    """
    Bybit API를 사용하여 오더북 데이터를 가져오는 함수.
    
    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")
        
    Returns:
        dict: 오더북 데이터 또는 None
    """
    try:
        # 올바른 함수명 사용: public_orderbook
        response = bybit.public_orderbook(symbol=symbol)
        logger.debug(f"get_order_book API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"오더북 데이터 조회 실패: {response['retMsg']}")
            return None

        order_book = response.get('result', {})
        if not order_book:
            logger.error("오더북 데이터가 비어 있습니다.")
            return None

        return order_book
    except Exception as e:
        logger.exception(f"get_order_book 함수에서 예외 발생: {e}")
        return None

def get_daily_ohlcv(bybit, symbol="BTCUSDT", interval="D", limit=100):
    """
    Bybit API를 사용하여 일별 OHLCV 데이터를 가져오는 함수.
    
    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")
        interval (str): 시간 간격 (기본값: "D" - 일별)
        limit (int): 가져올 데이터의 개수 (기본값: 100)
        
    Returns:
        pandas.DataFrame: OHLCV 데이터프레임 또는 None
    """
    try:
        response = bybit.query_kline(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        logger.debug(f"get_daily_ohlcv API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"OHLCV 데이터 조회 실패: {response['retMsg']}")
            return None

        ohlcv_data = response.get('result', {}).get('list', [])
        if not ohlcv_data:
            logger.error("OHLCV 데이터가 비어 있습니다.")
            return None

        # pandas DataFrame으로 변환
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['openTime'], unit='s')
        df.set_index('timestamp', inplace=True)

        # 기술 지표 추가 (예: 이동 평균)
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()

        logger.info("일별 OHLCV 데이터 조회 및 처리 완료.")
        return df
    except Exception as e:
        handle_error(e, "get_daily_ohlcv 함수")
        return None

def get_hourly_ohlcv(bybit, symbol="BTCUSDT", interval="60", limit=100):
    """
    Bybit API를 사용하여 시간별 OHLCV 데이터를 가져오는 함수.
    
    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")
        interval (str): 시간 간격 (기본값: "60" - 시간별)
        limit (int): 가져올 데이터의 개수 (기본값: 100)
        
    Returns:
        pandas.DataFrame: OHLCV 데이터프레임 또는 None
    """
    try:
        response = bybit.query_kline(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        logger.debug(f"get_hourly_ohlcv API 응답: {response}")

        if response['retCode'] != 0:
            logger.error(f"OHLCV 데이터 조회 실패: {response['retMsg']}")
            return None

        ohlcv_data = response.get('result', {}).get('list', [])
        if not ohlcv_data:
            logger.error("OHLCV 데이터가 비어 있습니다.")
            return None

        # pandas DataFrame으로 변환
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['openTime'], unit='s')
        df.set_index('timestamp', inplace=True)

        # 기술 지표 추가 (예: RSI)
        df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        logger.info("시간별 OHLCV 데이터 조회 및 처리 완료.")
        return df
    except Exception as e:
        handle_error(e, "get_hourly_ohlcv 함수")
        return None

def request_ai_trading_decision(collection, balance_data, daily_ohlcv, hourly_ohlcv):
    """
    OpenAI GPT-4 API를 호출하여 AI의 트레이딩 결정과 진입 퍼센트를 생성하는 함수.
    
    Parameters:
        collection (pymongo.collection.Collection): MongoDB 컬렉션 객체
        balance_data (dict): 현재 잔고 데이터
        daily_ohlcv (pd.DataFrame): 일별 OHLCV 데이터프레임
        hourly_ohlcv (pd.DataFrame): 시간별 OHLCV 데이터프레임
    
    Returns:
        dict: AI의 트레이딩 결정 (예: {'action': 'long', 'entry_percentage': 10})
    """
    try:
        # 최근 7일간의 거래 기록 조회
        seven_days_ago = datetime.utcnow() - pd.Timedelta(days=7)
        cursor = collection.find({"timestamp": {"$gte": seven_days_ago.isoformat()}}, {"_id": 0})
        trades = list(cursor)
        trades_df = pd.DataFrame(trades)

        # 현재 시장 데이터 준비 (예시: 마지막 시간별 OHLCV 데이터)
        current_market_data = hourly_ohlcv.tail(1).to_json(orient='records')

        # 전체 성과 계산 (예시: 최근 7일간의 수익률 합계)
        performance = trades_df['profit'].sum() if 'profit' in trades_df.columns else 0.0

        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions. Based on the analysis, suggest one of the following actions: 'long', 'short', 'close_long', 'close_short', 'hold'. Also, provide the percentage of the total capital to be used for entering a position when applicable."
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

Based on your analysis, suggest one of the following actions: 'long', 'short', 'close_long', 'close_short', 'hold'.
If the action is 'long' or 'short', also provide the entry percentage as a number between 1 and 100.

Provide your response in the following JSON format:
{
    "action": "long",
    "entry_percentage": 10
}
Limit your response to 250 words or less.
"""
                }
            ]
        )

        try:
            response_content = response.choices[0].message.content
            logger.debug(f"AI 응답 내용: {response_content}")
            logger.info("AI 트레이딩 결정 요청 성공.")
            return parse_trading_decision(response_content)
        except (IndexError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"Error extracting response content: {e}")
            return None

    except Exception as e:
        handle_error(e, "request_ai_trading_decision 함수")
        return None

def parse_trading_decision(response_content):
    """
    AI의 응답에서 트레이딩 결정을 파싱하는 함수.
    
    Parameters:
        response_content (str): AI의 응답 내용
    
    Returns:
        dict: 트레이딩 결정 (예: {'action': 'long', 'entry_percentage': 10})
    """
    try:
        # JSON 형식으로 응답을 파싱
        # AI가 제공한 JSON 블록 추출
        json_match = re.search(r'\{.*?\}', response_content, re.DOTALL)
        if not json_match:
            logger.warning("AI 응답에서 JSON 블록을 찾을 수 없습니다.")
            return {"action": "hold", "entry_percentage": 0}

        json_str = json_match.group(0)
        decision = json.loads(json_str)

        # 유효성 검사
        action = decision.get("action", "hold").lower()
        if action not in ["long", "short", "close_long", "close_short", "hold"]:
            logger.warning(f"알 수 없는 트레이딩 결정: {action}. 기본값으로 'hold'를 선택합니다.")
            action = "hold"

        entry_percentage = decision.get("entry_percentage", None)
        if action in ["long", "short"]:
            if entry_percentage is None:
                logger.warning("롱 또는 숏 액션에 대해 entry_percentage가 제공되지 않았습니다. 기본값으로 10%를 사용합니다.")
                entry_percentage = 10
            else:
                # 퍼센트 유효성 검사
                try:
                    entry_percentage = float(entry_percentage)
                    if not (1 <= entry_percentage <= 100):
                        logger.warning(f"entry_percentage가 유효하지 않습니다: {entry_percentage}. 기본값으로 10%를 사용합니다.")
                        entry_percentage = 10
                except ValueError:
                    logger.warning(f"entry_percentage가 숫자가 아닙니다: {entry_percentage}. 기본값으로 10%를 사용합니다.")
                    entry_percentage = 10
        else:
            entry_percentage = 0  # 롱 청산, 숏 청산, 보유 시에는 필요 없음

        logger.info(f"AI 트레이딩 결정: {action}, Entry Percentage: {entry_percentage}%")
        return {"action": action, "entry_percentage": entry_percentage}
    except json.JSONDecodeError as e:
        logger.error(f"JSON 디코딩 오류: {e}")
        return {"action": "hold", "entry_percentage": 0}
    except Exception as e:
        handle_error(e, "parse_trading_decision 함수")
        return {"action": "hold", "entry_percentage": 0}

def execute_trading_decision(bybit, collection, trading_decision, balance_data, current_position):
    """
    트레이딩 결정을 실행하는 함수.
    
    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        collection (pymongo.collection.Collection): MongoDB 컬렉션 객체
        trading_decision (dict): AI의 트레이딩 결정
        balance_data (dict): 현재 잔고 데이터
        current_position (str): 현재 포지션 ('long', 'short', 'none')
    """
    try:
        action = trading_decision.get("action", "hold")
        entry_percentage = trading_decision.get("entry_percentage", 0)
        equity = balance_data.get("equity", 0)

        logger.info(f"트레이딩 결정: {action}, Entry Percentage: {entry_percentage}%")

        if action == "long":
            if current_position == "none":
                # 롱 포지션 열기
                qty = calculate_order_qty(equity, entry_percentage, price=None)
                if qty <= 0:
                    logger.error("계산된 주문 수량이 유효하지 않습니다.")
                    return
                order = bybit.place_active_order(
                    symbol="BTCUSDT",
                    side="Buy",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel"
                )
                logger.info(f"롱 주문 실행: {order}")
                # 주문 결과를 MongoDB에 기록
                log_trade(collection, "long", qty, order.get("price", 0))
            else:
                logger.info("이미 포지션을 보유 중입니다. 롱 포지션을 열지 않습니다.")

        elif action == "short":
            if current_position == "none":
                # 숏 포지션 열기
                qty = calculate_order_qty(equity, entry_percentage, price=None)
                if qty <= 0:
                    logger.error("계산된 주문 수량이 유효하지 않습니다.")
                    return
                order = bybit.place_active_order(
                    symbol="BTCUSDT",
                    side="Sell",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel"
                )
                logger.info(f"숏 주문 실행: {order}")
                # 주문 결과를 MongoDB에 기록
                log_trade(collection, "short", qty, order.get("price", 0))
            else:
                logger.info("이미 포지션을 보유 중입니다. 숏 포지션을 열지 않습니다.")

        elif action == "close_long":
            if current_position == "long":
                # 롱 포지션 청산
                qty = get_position_qty(bybit, symbol="BTCUSDT")
                if qty is None or qty <= 0:
                    logger.error("포지션 청산을 위한 수량을 가져오지 못했습니다.")
                    return
                order = bybit.place_active_order(
                    symbol="BTCUSDT",
                    side="Sell",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel"
                )
                logger.info(f"롱 포지션 청산 주문 실행: {order}")
                # 주문 결과를 MongoDB에 기록
                log_trade(collection, "close_long", qty, order.get("price", 0))
            else:
                logger.info("롱 포지션을 보유하고 있지 않습니다. 청산을 수행하지 않습니다.")

        elif action == "close_short":
            if current_position == "short":
                # 숏 포지션 청산
                qty = get_position_qty(bybit, symbol="BTCUSDT")
                if qty is None or qty <= 0:
                    logger.error("포지션 청산을 위한 수량을 가져오지 못했습니다.")
                    return
                order = bybit.place_active_order(
                    symbol="BTCUSDT",
                    side="Buy",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel"
                )
                logger.info(f"숏 포지션 청산 주문 실행: {order}")
                # 주문 결과를 MongoDB에 기록
                log_trade(collection, "close_short", qty, order.get("price", 0))
            else:
                logger.info("숏 포지션을 보유하고 있지 않습니다. 청산을 수행하지 않습니다.")

        elif action == "hold":
            logger.info("트레이딩 결정: 보유 (Hold). 아무 작업도 수행하지 않습니다.")

        else:
            logger.warning(f"알 수 없는 트레이딩 결정: {action}. 아무 작업도 수행하지 않습니다.")

    except Exception as e:
        handle_error(e, "execute_trading_decision 함수")

def calculate_order_qty(equity, entry_percentage, price):
    """
    전체 자본에서 퍼센트에 따라 주문 수량을 계산하는 함수.
    
    Parameters:
        equity (float): 전체 자본
        entry_percentage (float): 자본 대비 퍼센트 (1-100)
        price (float): 현재 가격 (시장가 주문 시 필요 없음)
        
    Returns:
        float: 주문 수량
    """
    try:
        investment = equity * (entry_percentage / 100)
        if price:
            qty = investment / price
        else:
            # 시장가 주문 시 최소 수량을 설정 (예시: 0.001 BTC)
            qty = 0.001
        # Bybit API에서 요구하는 최소 주문 수량 및 단위 확인 필요
        # 예시로, 0.001 BTC 단위로 설정
        qty = round(qty, 6)
        return qty
    except Exception as e:
        handle_error(e, "calculate_order_qty 함수")
        return 0.0

def get_position_qty(bybit, symbol="BTCUSDT"):
    """
    현재 포지션의 수량을 조회하는 함수.
    
    Parameters:
        bybit (HTTP): Bybit API 클라이언트 객체
        symbol (str): 심볼 이름 (기본값: "BTCUSDT")
        
    Returns:
        float: 현재 포지션 수량 또는 None
    """
    try:
        response = bybit.my_position(symbol=symbol)
        logger.debug(f"get_position_qty API 응답: {response}")
        
        if response['retCode'] != 0:
            logger.error(f"포지션 수량 조회 실패: {response['retMsg']}")
            return None
        
        positions = response.get('result', [])
        if not positions:
            logger.info("현재 포지션이 없습니다.")
            return None
        
        position = positions[0]
        qty = float(position.get('size', 0))
        return qty
    except Exception as e:
        handle_error(e, "get_position_qty 함수")
        return None

def log_trade(collection, trade_type, amount, price):
    """
    트레이딩 기록을 MongoDB에 저장하는 함수.
    
    Parameters:
        collection (pymongo.collection.Collection): MongoDB 컬렉션 객체
        trade_type (str): 거래 유형 ('long', 'short', 'close_long', 'close_short')
        amount (float): 거래 수량
        price (float): 거래 가격
    """
    trade_record = {
        "timestamp": get_current_timestamp(),
        "trade_type": trade_type,
        "amount": amount,
        "price": price
    }
    try:
        collection.insert_one(trade_record)
        logger.info(f"{trade_type.capitalize()} 거래가 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        handle_error(e, "log_trade 함수")

if __name__ == "__main__":
    # MongoDB, Bybit 및 OpenAI 연결 설정
    trades_collection = setup_mongodb()
    bybit = setup_bybit()
    setup_openai()

    # 트레이딩 작업을 테스트하기 위해 job 함수 호출
    job(bybit, trades_collection)
