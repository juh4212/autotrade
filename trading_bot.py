import os
import time
import logging
import traceback
import requests
from pymongo import MongoClient
from pybit import HTTP
import openai
import schedule

# 환경 변수 로드
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 설정하여 상세 로그 기록
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # 표준 출력으로 로그 전송 (Docker 로그로 캡처됨)
    ]
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
        logger.debug(traceback.format_exc())
        raise

# Bybit 클라이언트 설정
def setup_bybit():
    try:
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.critical("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
            raise ValueError("BYBIT_API_KEY 또는 BYBIT_API_SECRET 환경 변수가 설정되지 않았습니다.")
        bybit = HTTP("https://api.bybit.com", api_key=api_key, api_secret=api_secret)
        logger.debug("Bybit API에 성공적으로 연결되었습니다.")
        return bybit
    except Exception as e:
        logger.critical(f"Bybit API 연결 오류: {e}")
        logger.debug(traceback.format_exc())
        raise

# OpenAI 설정
def setup_openai():
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.critical("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        openai.api_key = openai_api_key
        logger.debug("OpenAI API 설정 완료.")
    except Exception as e:
        logger.critical(f"OpenAI API 설정 오류: {e}")
        logger.debug(traceback.format_exc())
        raise

# 거래 기록을 DB에 저장하기
def log_trade(collection, decision, percentage, reason, btc_balance, usdt_balance, btc_avg_buy_price, current_btc_price, reflection):
    try:
        trade = {
            "timestamp": time.time(),
            "decision": decision,
            "percentage": percentage,
            "reason": reason,
            "btc_balance": btc_balance,
            "usdt_balance": usdt_balance,
            "btc_avg_buy_price": btc_avg_buy_price,
            "btc_usdt_price": current_btc_price,
            "reflection": reflection
        }
        collection.insert_one(trade)
        logger.debug("거래 기록이 MongoDB에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"거래 기록 저장 오류: {e}")
        logger.debug(traceback.format_exc())

# Fear and Greed Index 가져오기
def get_fear_and_greed_index():
    try:
        logger.debug("Fear and Greed Index 가져오기 시작...")
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        data = response.json()
        value = data['data'][0]['value']
        logger.debug(f"Fear and Greed Index: {value}")
        return value
    except requests.RequestException as e:
        logger.error(f"Fear and Greed Index 가져오기 오류: {e}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        logger.debug("get_fear_and_greed_index 함수 실행 완료.")

# Bitcoin 뉴스 가져오기
def get_bitcoin_news():
    try:
        logger.debug("Bitcoin 뉴스 가져오기 시작...")
        news_api_key = os.getenv("NEWS_API_KEY")
        if not news_api_key:
            logger.error("NEWS_API_KEY 환경 변수가 설정되지 않았습니다.")
            return []
        url = f"https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&apiKey={news_api_key}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        headlines = [article['title'] for article in articles]
        logger.debug(f"{len(headlines)}개의 뉴스 헤드라인을 가져왔습니다.")
        return headlines
    except requests.RequestException as e:
        logger.error(f"Bitcoin 뉴스 가져오기 오류: {e}")
        logger.debug(traceback.format_exc())
        return []
    finally:
        logger.debug("get_bitcoin_news 함수 실행 완료.")

# AI를 사용한 트레이딩 결정
def ai_trading(collection, bybit):
    try:
        logger.debug("AI 트레이딩 결정 시작...")
        # Fear and Greed Index 가져오기
        fng = get_fear_and_greed_index()
        
        # Bitcoin 뉴스 가져오기
        news_headlines = get_bitcoin_news()
        
        # 현재 잔고 조회
        balance_info = bybit.get_wallet_balance()
        if balance_info and 'result' in balance_info:
            balances = balance_info['result']
            btc_balance = 0
            usdt_balance = 0
            btc_avg_buy_price = 0
            for balance in balances:
                if balance['coin'] == 'BTC':
                    btc_balance = float(balance['wallet_balance'])
                    btc_avg_buy_price = float(balance.get('avgPrice', 0))
                    logger.debug(f"BTC 잔고: {btc_balance}, 평균 매수가: {btc_avg_buy_price}")
                elif balance['coin'] == 'USDT':
                    usdt_balance = float(balance['wallet_balance'])
                    logger.debug(f"USDT 잔고: {usdt_balance}")
            logger.debug(f"현재 BTC 잔고: {btc_balance}, USDT 잔고: {usdt_balance}")
        else:
            logger.error("잔고 정보를 가져올 수 없습니다.")
            return
        
        # 현재 BTC 가격 조회
        btc_price_info = bybit.latest_information_for_symbol(symbol="BTCUSDT")
        if btc_price_info and 'result' in btc_price_info:
            current_btc_price = float(btc_price_info['result'][0]['last_price'])
            logger.debug(f"현재 BTC 가격: {current_btc_price} USDT")
        else:
            logger.error("현재 BTC 가격을 가져올 수 없습니다.")
            return
        
        # OpenAI를 사용하여 트레이딩 결정 생성
        prompt = (
            f"현재 BTC 잔고는 {btc_balance} BTC이고, USDT 잔고는 {usdt_balance} USDT입니다.\n"
            f"평균 매수가: {btc_avg_buy_price} USDT\n"
            f"현재 BTC 가격: {current_btc_price} USDT\n"
            f"Fear and Greed Index: {fng}\n"
            f"Bitcoin 뉴스 헤드라인: {news_headlines}\n"
            f"위 정보를 바탕으로 'buy', 'sell', 'hold' 중 하나의 결정을 내려주세요."
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful trading assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.5,
            )
            decision = response.choices[0].message['content'].strip()
            logger.debug(f"AI Decision: {decision}")
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            logger.debug(traceback.format_exc())
            return
        
        # 거래 결정에 따른 행동 수행
        if decision.lower() == "buy":
            percentage = 5  # 예시로 5% 매수
            amount = (usdt_balance * (percentage / 100)) / current_btc_price
            try:
                buy_order = bybit.place_active_order(
                    symbol="BTCUSDT",
                    side="Buy",
                    order_type="Market",
                    qty=amount,
                    time_in_force="GoodTillCancel"
                )
                logger.info(f"Buy order executed: {buy_order}")
                reason = "AI decision to buy based on market analysis."
                reflection = "Successfully executed buy order."
                log_trade(collection, "BUY", percentage, reason, btc_balance, usdt_balance, btc_avg_buy_price, current_btc_price, reflection)
            except Exception as e:
                logger.error(f"Buy order 실행 오류: {e}")
                logger.debug(traceback.format_exc())
        
        elif decision.lower() == "sell":
            percentage = 5  # 예시로 5% 매도
            amount = btc_balance * (percentage / 100)
            if amount > 0:
                try:
                    sell_order = bybit.place_active_order(
                        symbol="BTCUSDT",
                        side="Sell",
                        order_type="Market",
                        qty=amount,
                        time_in_force="GoodTillCancel"
                    )
                    logger.info(f"Sell order executed: {sell_order}")
                    reason = "AI decision to sell based on market analysis."
                    reflection = "Successfully executed sell order."
                    log_trade(collection, "SELL", percentage, reason, btc_balance, usdt_balance, btc_avg_buy_price, current_btc_price, reflection)
                except Exception as e:
                    logger.error(f"Sell order 실행 오류: {e}")
                    logger.debug(traceback.format_exc())
            else:
                logger.warning("매도할 BTC 잔고가 부족합니다.")
        
        elif decision.lower() == "hold":
            logger.info("Hold 결정: 현재 포지션을 유지합니다.")
            reason = "AI decision to hold based on market analysis."
            reflection = "No action taken; holding current position."
            log_trade(collection, "HOLD", 0, reason, btc_balance, usdt_balance, btc_avg_buy_price, current_btc_price, reflection)
        
        else:
            logger.error(f"유효하지 않은 결정: {decision}")
    
    # 트레이딩 봇 주기적 실행 함수
    def run_trading_bot(trades_collection, bybit):
        try:
            ai_trading(trades_collection, bybit)
        except Exception as e:
            logger.error(f"트레이딩 봇 실행 오류: {e}")
            logger.debug(traceback.format_exc())

    if __name__ == "__main__":
        try:
            # 설정 초기화
            trades_collection = setup_mongodb()
            bybit = setup_bybit()
            setup_openai()
            
            # 트레이딩 봇 스케줄링 (예: 매 4시간마다 실행)
            schedule.every(4).hours.do(run_trading_bot, trades_collection, bybit)
            logger.info("트레이딩 봇 스케줄러 시작: 매 4시간마다 실행됩니다.")
            
            while True:
                schedule.run_pending()
                time.sleep(1)
        except Exception as e:
            logger.critical(f"트레이딩 봇 초기화 실패: {e}")
            logger.debug(traceback.format_exc())
            exit(1)
