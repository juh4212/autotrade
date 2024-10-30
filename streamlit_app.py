import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import pyupbit  # 현재 BTC 가격 가져오기 위해 추가

# 데이터베이스 연결 함수
def get_connection():
    client = MongoClient('mongodb://localhost:27017/')  # MongoDB 연결
    db = client['bitcoin_trades_db']  # 데이터베이스 선택
    collection = db['trades']  # 컬렉션 선택
    return collection

# 데이터 로드 함수
def load_data():
    collection = get_connection()
    data = list(collection.find())
    df = pd.DataFrame(data)
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])  # MongoDB의 기본 '_id' 컬럼 제거
    return df

# 초기 투자 금액 계산 함수
def calculate_initial_investment(df):
    initial_krw_balance = df.iloc[0]['krw_balance']
    initial_btc_balance = df.iloc[0]['btc_balance']
    initial_btc_price = df.iloc[0]['btc_krw_price']
    initial_total_investment = initial_krw_balance + (initial_btc_balance * initial_btc_price)
    return initial_total_investment

# 현재 투자 금액 계산 함수
def calculate_current_investment(df):
    current_krw_balance = df.iloc[-1]['krw_balance']
    current_btc_balance = df.iloc[-1]['btc_balance']
    current_btc_price = pyupbit.get_current_price("KRW-BTC")  # 현재 BTC 가격 가져오기
    current_total_investment = current_krw_balance + (current_btc_balance * current_btc_price)
    return current_total_investment

# 메인 함수
def main():
    st.title('Bitcoin Trades Viewer')

    # 데이터 로드
    df = load_data()

    if df.empty:
        st.warning('No trade data available.')
        return

    # 데이터 타입 변환 (필요한 경우)
    numeric_columns = ['krw_balance', 'btc_balance', 'btc_avg_buy_price', 'btc_krw_price', 'percentage']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 날짜 형식 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 초기 투자 금액 계산
    initial_investment = calculate_initial_investment(df)

    # 현재 투자 금액 계산
    current_investment = calculate_current_investment(df)

    # 수익률 계산
    profit_rate = ((current_investment - initial_investment) / initial_investment) * 100

    # 수익률 표시
    st.header(f'📈 Current Profit Rate: {profit_rate:.2f}%')

    # 기본 통계
    st.header('기본 통계')
    st.write(f"총 거래 횟수: {len(df)}")
    st.write(f"첫 거래 날짜: {df['timestamp'].min()}")
    st.write(f"마지막 거래 날짜: {df['timestamp'].max()}")

    # 거래 내역 표시
    st.header('거래 내역')
    st.dataframe(df)

    # 거래 결정 분포
    st.header('거래 결정 분포')
    decision_counts = df['decision'].value_counts()
    if not decision_counts.empty:
        fig = px.pie(values=decision_counts.values, names=decision_counts.index, title='거래 결정')
        st.plotly_chart(fig)
    else:
        st.write("표시할 거래 결정이 없습니다.")

    # BTC 잔액 변화
    st.header('BTC 잔액 변화')
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC 잔액')
    st.plotly_chart(fig)

    # KRW 잔액 변화
    st.header('KRW 잔액 변화')
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW 잔액')
    st.plotly_chart(fig)

    # BTC 평균 매수가 변화
    st.header('BTC 평균 매수가 변화')
    fig = px.line(df, x='timestamp', y='btc_avg_buy_price', title='BTC 평균 매수가')
    st.plotly_chart(fig)

    # BTC 가격 변화
    st.header('BTC 가격 변화')
    fig = px.line(df, x='timestamp', y='btc_krw_price', title='BTC 가격 (KRW)')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
