import os
import time
import logging
import requests
from pymongo import MongoClient
from pybit.usdt_perpetual import HTTP
import openai
import schedule
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import json
import re

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# MongoDB 설정
def setup_mongodb():
    try:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            logger.critical("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
            raise ValueError("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
        client = MongoClient(mongo_uri)
        db = client['bitcoin_trades_db']
        trades_collection = db['trades']
        logger.debug("MongoDB에 성공적으로 연결되었습니다.")
        return trades_collection
    except Exception as e:
        logger.critical(f"MongoDB 연결 오류: {e}")
        raise

# Bybit 선물 거래 클라이언트 설정
def setup_bybit():
    try:
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.critical("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
            raise ValueError("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
        bybit = HTTP("https://api.bybit.com", api_key=api_key, api_secret=api_secret)
        logger.debug("Bybit USDT Perpetual API에 성공적으로 연결되었습니다.")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        raise

# OpenAI 설정
def setup_openai():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.critical("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    openai.api_key = openai_api_key
    logger.debug("OpenAI API 설정 완료.")

# 거래 기록을 DB에 저장하기
def log_trade(collection, decision, percentage, reason, btc_balance, usdt_balance, btc_avg_buy_price, btc_usdt_price, reflection):
    try:
        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "percentage": percentage,
            "reason": reason,
            "btc_balance": btc_balance,
            "usdt_balance": usdt_balance,
            "btc_avg_buy_price": btc_avg_buy_price,
            "btc_usdt_price": btc_usdt_price,
            "reflection": reflection
        }
        collection.insert_one(trade)
        logger.info("거래 기록이 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"거래 기록 저장 오류: {e}")

# 최근 거래 기록을 조회하는 함수
def get_recent_trades(collection, days=7):
    seven_days_ago = datetime.utcnow() - timedelta(days=days)
    try:
        trades = list(collection.find({"timestamp": {"$gt": seven_days_ago.isoformat()}}).sort("timestamp", -1))
        if not trades:
            logger.info("최근 거래 기록이 없습니다.")
            return pd.DataFrame()
        return pd.DataFrame(trades)
    except Exception as e:
        logger.error(f"최근 거래 기록 조회 오류: {e}")
        return pd.DataFrame()

# Fear and Greed Index 가져오기
def get_fear_and_greed_index():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        data = response.json()
        value = data['data'][0]['value']
        logger.info(f"Fear and Greed Index: {value}")
        return value
    except requests.RequestException as e:
        logger.error(f"Fear and Greed Index 가져오기 오류: {e}")
        return None

# Bitcoin 뉴스 가져오기
def get_bitcoin_news():
    try:
        news_api_key = os.getenv("NEWS_API_KEY")
        if not news_api_key:
            logger.error("NEWS_API_KEY 환경 변수가 설정되지 않았습니다.")
            return []
        url = f"https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&apiKey={news_api_key}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        headlines = [article['title'] for article in articles]
        logger.info(f"{len(headlines)}개의 뉴스 헤드라인을 가져왔습니다.")
        return headlines
    except requests.RequestException as e:
        logger.error(f"Bitcoin 뉴스 가져오기 오류: {e}")
        return []

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."},
                {"role": "user", "content": f"Recent trading data: {trades_df.to_json(orient='records')}\n\nCurrent market data: {current_market_data}\n\nOverall performance in the last 7 days: {performance:.2f}%"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"OpenAI reflection 생성 오류: {e}")
        return None

# AI 트레이딩 로직
def ai_trading(trades_collection, bybit):
    try:
        fng = get_fear_and_greed_index()
        news_headlines = get_bitcoin_news()
        
        # 잔고 조회
        wallet_balance = bybit.get_wallet_balance()["result"]["USDT"]["wallet_balance"]
        btc_position = bybit.get_positions(symbol="BTCUSDT")["result"][0]
        btc_balance = btc_position["size"]
        entry_price = btc_position["entry_price"]
        logger.info(f"BTC 잔고: {btc_balance}, USDT 잔고: {wallet_balance}, 평균 매수가: {entry_price}")

        # OpenAI를 사용하여 AI 결정 생성
        decision = "hold"  # 기본 값
        try:
            prompt = f"Fear and Greed Index: {fng}, BTC balance: {btc_balance}, USDT balance: {wallet_balance}"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            decision = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"AI 결정 생성 오류: {e}")

        # 거래 수행
        percentage = 5
        if decision.lower() == "buy":
            amount = (wallet_balance * (percentage / 100)) / entry_price
            bybit.place_active_order(
                symbol="BTCUSDT",
                side="Buy",
                order_type="Market",
                qty=amount,
                time_in_force="GoodTillCancel"
            )
            log_trade(trades_collection, "BUY", percentage, "AI decision to buy", btc_balance, wallet_balance, entry_price, entry_price, "Buy order placed")

        elif decision.lower() == "sell":
            amount = btc_balance * (percentage / 100)
            bybit.place_active_order(
                symbol="BTCUSDT",
                side="Sell",
                order_type="Market",
                qty=amount,
                time_in_force="GoodTillCancel"
            )
            log_trade(trades_collection, "SELL", percentage, "AI decision to sell", btc_balance, wallet_balance, entry_price, entry_price, "Sell order placed")
        
        logger.info(f"AI 결정: {decision}")
    except Exception as e:
        logger.error(f"트레이딩 실행 오류: {e}")

# 주기적으로 트레이딩을 실행하는 함수
def run_trading_bot():
    try:
        trades_collection = setup_mongodb()
        bybit = setup_bybit()
        setup_openai()

        schedule.every(4).hours.do(ai_trading, trades_collection, bybit)
        logger.info("트레이딩 봇 스케줄러 시작: 매 4시간마다 실행됩니다.")

        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logger.critical(f"트레이딩 봇 초기화 실패: {e}")
        exit(1)

if __name__ == "__main__":
    run_trading_bot()
