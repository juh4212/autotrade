# ai_judgment.py

import os
import re
import json
import logging
import openai  # OpenAI 라이브러리 임포트
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 필요 시 DEBUG로 변경
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def calculate_performance(trades_df):
    """
    최근 거래 내역을 기반으로 퍼포먼스를 계산하는 함수
    """
    if trades_df.empty:
        return 0.0
    initial_balance = trades_df.iloc[0]['balance']
    final_balance = trades_df.iloc[-1]['balance']
    performance = ((final_balance - initial_balance) / initial_balance) * 100
    return performance

def generate_reflection(trades_df, current_market_data):
    """
    AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
    """
    performance = calculate_performance(trades_df)  # 투자 퍼포먼스 계산

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("OpenAI API key is missing or invalid.")
        return None

    openai.api_key = openai_api_key

    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 실제 사용 중인 모델로 변경
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
{json.dumps(current_market_data)}

Overall performance in the last 7 days: {performance:.2f}%

Please analyze this data and provide:
1. A brief reflection on the recent trading decisions
2. Insights on what worked well and what didn't
3. Suggestions for improvement in future trading decisions
4. Any patterns or trends you notice in the market data

Limit your response to 250 words or less.
"""
            }
        ]
    )

    try:
        response_content = response.choices[0].message.content
        return response_content
    except (IndexError, AttributeError) as e:
        logging.error(f"Error extracting response content: {e}")
        return None

def get_ai_decision(trades_df, current_market_data):
    """
    AI의 판단을 받아오는 함수
    """
    reflection = generate_reflection(trades_df, current_market_data)
    if not reflection:
        logging.error("Failed to generate reflection from AI.")
        return None

    logging.info(f"AI Reflection: {reflection}")

    # AI로부터 매매 결정을 생성
    decision = analyze_reflection(reflection)
    return decision

def analyze_reflection(reflection_text):
    """
    AI의 반성을 분석하여 매매 결정을 도출하는 함수
    """
    try:
        # AI의 반성 텍스트에서 JSON 부분 추출
        json_match = re.search(r'\{.*?\}', reflection_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            decision = parsed_json.get('decision')
            percentage = parsed_json.get('percentage')
            reason = parsed_json.get('reason')
            return {'decision': decision, 'percentage': percentage, 'reason': reason}
        else:
            logging.error("No JSON found in AI reflection.")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return None
