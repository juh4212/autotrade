# scheduler.py

import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import time
from data_collection import get_market_data, get_order_history, get_fear_greed_index
from data_processing import add_technical_indicators, analyze_recent_trades
from ai_judgment import get_ai_decision
from trade_execution import place_order
from record_storage import save_trade_record, save_investment_performance
from reflection_improvement import get_reflection_and_improvement, apply_improvements
from discord_bot import notify_discord

# 로그 디렉토리 생성 및 권한 설정
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
os.chmod(LOG_DIR, 0o777)  # 개발 단계에서는 777, 운영 환경에서는 최소 권한 부여

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'trading_bot.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def job():
    try:
        logging.info("작업 시작")

        # 데이터 수집
        df = get_market_data()
        if df.empty:
            logging.warning("시장 데이터가 비어 있습니다.")
            notify_discord("시장 데이터가 비어 있습니다.")
            return
        df = add_technical_indicators(df)
        access_key = os.getenv('BYBIT_API_KEY')
        secret_key = os.getenv('BYBIT_API_SECRET')
        symbol = "BTCUSD"
        orders = get_order_history(symbol=symbol, limit=100)
        analysis = analyze_recent_trades(orders)
        fg_index = get_fear_greed_index()

        # AI 판단
        decision, reason = get_ai_decision(df, analysis, fg_index)
        logging.info(f"결정: {decision} - 이유: {reason}")

        # Discord 알림
        if decision in ["buy", "sell"]:
            message = f"매매 신호: {decision.upper()} - 이유: {reason}"
            notify_discord(message)
        else:
            message = "현재 포지션을 유지합니다. 관망 중."
            notify_discord(message)

        # 거래 실행
        if decision == "buy":
            qty = 1  # 구매할 수량 설정 (예: 1 BTC)
            order = place_order(symbol, "Buy", qty=qty, order_type="Market")
        elif decision == "sell":
            qty = 1  # 판매할 수량 설정 (예: 1 BTC)
            order = place_order(symbol, "Sell", qty=qty, order_type="Market")
        else:
            order = None

        # 기록 저장
        if order and 'result' in order:
            save_trade_record(
                symbol=symbol,
                side="Buy" if decision == "buy" else "Sell",
                qty=qty if decision in ["buy", "sell"] else None,
                price=order['result']['price'] if 'price' in order['result'] else None,
                order_type=order['result']['order_type'] if 'order_type' in order['result'] else "Market",
                status=order['result']['order_status'] if 'order_status' in order['result'] else "Unknown",
                response=order
            )

        # 투자 성과 저장
        save_investment_performance(
            total_trades=analysis['total_trades'],
            profitable_trades=analysis['profitable_trades'],
            win_rate=analysis['win_rate']
        )

        # 반성 및 개선
        reflection = get_reflection_and_improvement(analysis)
        logging.info(f"반성 및 개선점: {reflection}")
        notify_discord(f"반성 및 개선점:\n{reflection}")
        apply_improvements(reflection)

        logging.info("작업 완료")

    except Exception as e:
        error_message = f"작업 중 에러 발생: {e}"
        logging.error(error_message)
        notify_discord(error_message)

def scheduler_job():
    scheduler = BackgroundScheduler()
    # 예: 매일 오전 9시에 실행
    scheduler.add_job(job, 'cron', hour=9, minute=0)
    scheduler.start()
    logging.info("스케줄러 시작")

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("스케줄러 종료")

# 테스트용 호출
if __name__ == "__main__":
    scheduler_job()
