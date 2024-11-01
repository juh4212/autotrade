# reflection_improvement.py

import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def get_reflection_and_improvement(trade_analysis):
    """
    투자 성과를 기반으로 AI에게 반성과 개선점을 요청합니다.
    """
    try:
        prompt = f"""
        다음은 최근 투자 성과입니다:
        - 총 거래 수: {trade_analysis['total_trades']}
        - 수익성 있는 거래 수: {trade_analysis['profitable_trades']}
        - 승률: {trade_analysis['win_rate']:.2f}%

        이 데이터를 바탕으로 투자 전략에 대한 반성과 향후 개선점을 제안해 주세요.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 암호화폐 투자 전략을 분석하고 개선점을 제안하는 AI입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )

        reflection = response.choices[0].message['content'].strip()
        return reflection
    except Exception as e:
        print(f"반성 및 개선점 도출 에러: {e}")
        return "반성 및 개선점 도출 실패."

def apply_improvements(reflection):
    """
    AI가 제안한 개선 사항을 적용합니다.
    (이 예제에서는 단순히 출력하지만, 실제 구현 시 자동화된 로직 추가 가능)
    """
    try:
        print(f"적용할 개선 사항:\n{reflection}")
        # TODO: 전략 수정 로직 구현
    except Exception as e:
        print(f"개선 사항 적용 에러: {e}")

# 테스트용 호출
if __name__ == "__main__":
    trade_analysis = {
        "total_trades": 10,
        "profitable_trades": 7,
        "win_rate": 70.0
    }
    reflection = get_reflection_and_improvement(trade_analysis)
    print(f"반성 및 개선점:\n{reflection}")

    apply_improvements(reflection)
