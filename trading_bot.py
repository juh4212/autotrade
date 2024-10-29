import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP  # Bybit v5 API를 사용 중임을 가정
import pandas as pd
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

if __name__ == "__main__":
    # MongoDB와 Bybit 연결 설정
    trades_collection = setup_mongodb()
    bybit = setup_bybit()

    # Bybit API를 사용하여 계좌 잔고 가져오기
    balance_data = get_account_balance(bybit)
    if balance_data:
        # MongoDB에 잔고 기록
        log_balance_to_mongodb(trades_collection, balance_data)

    # 예시: 거래 기록 추가 (실제 거래 로직에 맞게 수정 필요)
    # decision, percentage, reason 등의 값은 실제 로직에 따라 설정
    # 여기서는 예시로 값을 넣었습니다.
    decision = "BUY"
    percentage = 10
    reason = "Market dip observed"
    btc_balance = 0.5
    krw_balance = 1000000  # 예시 금액
    btc_avg_buy_price = 50000
    btc_krw_price = 55000
    reflection = ""  # 나중에 AI가 채울 부분

    # 거래 기록 저장
    log_trade(trades_collection, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection)

    # 최근 거래 기록 조회
    recent_trades_df = get_recent_trades(trades_collection, days=7)

    if not recent_trades_df.empty:
        # 현재 시장 데이터 예시 (실제 데이터로 대체 필요)
        current_market_data = {
            "BTC_USD": 56000,
            "market_trend": "uptrend",
            "volume": 1200
        }

        # AI를 사용하여 반성 생성
        reflection_text = generate_reflection(recent_trades_df, current_market_data)
        if reflection_text:
            # 최신 거래에 반성 추가
            latest_trade = recent_trades_df.iloc[0]
            trade_id = latest_trade['_id']
            try:
                trades_collection.update_one(
                    {"_id": trade_id},
                    {"$set": {"reflection": reflection_text}}
                )
                logger.info("AI 반성이 최신 거래에 성공적으로 추가되었습니다.")
            except Exception as e:
                logger.error(f"AI 반성을 거래에 추가하는 중 오류 발생: {e}")
