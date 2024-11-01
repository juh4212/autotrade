# ai_judgment.py

import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def get_ai_decision(market_data, trade_analysis, fear_greed_index):
    """
    OpenAI GPT-4를 사용하여 투자 판단을 받습니다.
    """
    try:
        # 최신 데이터 추출
        latest_data = market_data.iloc[-1]
        prompt = f"""
        다음은 현재 비트코인 시장 데이터입니다:
        - 최근 20일 이동평균선: {latest_data['ma20']}
        - 최근 50일 이동평균선: {latest_data['ma50']}
        - RSI: {latest_data['rsi']}
        - MACD: {latest_data['macd']}, Signal: {latest_data['macd_signal']}
        - MACD Difference: {latest_data['macd_diff']}
        - 공포 탐욕 지수: {fear_greed_index}

        최근 거래 내역 분석:
        - 총 거래 수: {trade_analysis['total_trades']}
        - 수익성 있는 거래 수: {trade_analysis['profitable_trades']}
        - 승률: {trade_analysis['win_rate']:.2f}%

        이 데이터를 바탕으로 현재 포지션을 어떻게 조정하는 것이 좋을지 결정 (롱, 숏, 청산, 관망)하고, 그 이유를 설명해 주세요.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 암호화폐 자동매매를 지원하는 AI입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )

        decision_text = response.choices[0].message['content'].strip()

        # 간단한 파싱 로직 (필요 시 정규 표현식 사용)
        if "롱" in decision_text or "buy" in decision_text.lower():
            decision = "buy"
        elif "숏" in decision_text or "sell" in decision_text.lower():
            decision = "sell"
        elif "청산" in decision_text or "close" in decision_text.lower():
            decision = "close"
        else:
            decision = "hold"

        return decision, decision_text
    except Exception as e:
        print(f"AI 판단 에러: {e}")
        return "hold", "AI 판단 실패로 관망 중."

# 테스트용 호출
if __name__ == "__main__":
    from data_collection import get_market_data, get_order_history, get_fear_greed_index
    from data_processing import add_technical_indicators, analyze_recent_trades

    df = get_market_data()
    df = add_technical_indicators(df)
    orders = get_order_history()
    analysis = analyze_recent_trades(orders)
    fg_index = get_fear_greed_index()

    decision, reason = get_ai_decision(df, analysis, fg_index)
    print(f"결정: {decision}")
    print(f"이유: {reason}")
