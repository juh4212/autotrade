# data_processing.py

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

def add_technical_indicators(df):
    """
    시장 데이터에 기술적 지표(MA, RSI, MACD)를 추가합니다.
    """
    try:
        # 이동평균선 (20일, 50일)
        df['ma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['ma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()

        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        return df
    except Exception as e:
        print(f"기술적 지표 추가 에러: {e}")
        return df

def analyze_recent_trades(trade_history):
    """
    최근 거래 내역을 분석하여 총 거래 수, 수익성 있는 거래 수, 승률을 계산합니다.
    """
    try:
        if not trade_history:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0
            }

        df = pd.DataFrame(trade_history)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values(by='created_at', ascending=False)

        # Bybit의 주문 응답에는 'profit' 필드가 없으므로, 수익성 계산을 위해 추가 로직 필요
        # 예시: 체결 가격과 주문 타입을 기반으로 간단한 수익성 계산
        # 실제 구현 시, 포지션 관리 및 수익 계산 로직을 정교하게 설계해야 합니다.

        # 임시로 수익성 있는 거래 수를 랜덤으로 설정 (실제 수익 계산 로직 필요)
        # 여기서는 'price'가 특정 기준 이상일 경우 수익성 있다고 가정
        # 실제 거래 전략에 맞게 수정 필요
        profitable_trades = df[df['price'] > 50000]  # 예시 기준
        total_trades = len(df)
        profitable_count = len(profitable_trades)
        win_rate = (profitable_count / total_trades) * 100 if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_count,
            "win_rate": win_rate
        }
    except Exception as e:
        print(f"거래 내역 분석 에러: {e}")
        return {
            "total_trades": 0,
            "profitable_trades": 0,
            "win_rate": 0.0
        }

# 테스트용 호출
if __name__ == "__main__":
    from data_collection import get_market_data, get_order_history, get_fear_greed_index

    df = get_market_data()
    df = add_technical_indicators(df)
    print(df[['close', 'ma20', 'ma50', 'rsi', 'macd', 'macd_signal']].tail())

    orders = get_order_history()
    analysis = analyze_recent_trades(orders)
    print(analysis)
