import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import pyupbit  # Added to fetch current BTC price

# 데이터베이스 연결 함수
def get_connection():
    return sqlite3.connect('bitcoin_trades.db')

# 데이터 로드 함수
def load_data():
    conn = get_connection()
    query = "SELECT * FROM trades"
    df = pd.read_sql_query(query, conn)
    conn.close()
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

    # 초기 투자 금액 계산
    initial_investment = calculate_initial_investment(df)

    # 현재 투자 금액 계산
    current_investment = calculate_current_investment(df)

    # 수익률 계산
    profit_rate = ((current_investment - initial_investment) / initial_investment) * 100

    # 수익률 표시
    st.header(f'📈 Current Profit Rate: {profit_rate:.2f}%')

    # 기본 통계
    st.header('Basic Statistics')
    st.write(f"Total number of trades: {len(df)}")
    st.write(f"First trade date: {df['timestamp'].min()}")
    st.write(f"Last trade date: {df['timestamp'].max()}")

    # 거래 내역 표시
    st.header('Trade History')
    st.dataframe(df)

    # 거래 결정 분포
    st.header('Trade Decision Distribution')
    decision_counts = df['decision'].value_counts()
    if not decision_counts.empty:
        fig = px.pie(values=decision_counts.values, names=decision_counts.index, title='Trade Decisions')
        st.plotly_chart(fig)
    else:
        st.write("No trade decisions to display.")

    # BTC 잔액 변화
    st.header('BTC Balance Over Time')
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC Balance')
    st.plotly_chart(fig)

    # KRW 잔액 변화
    st.header('KRW Balance Over Time')
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW Balance')
    st.plotly_chart(fig)

    # BTC 평균 매수가 변화
    st.header('BTC Average Buy Price Over Time')
    fig = px.line(df, x='timestamp', y='btc_avg_buy_price', title='BTC Average Buy Price')
    st.plotly_chart(fig)

    # BTC 가격 변화
    st.header('BTC Price Over Time')
    fig = px.line(df, x='timestamp', y='btc_krw_price', title='BTC Price (KRW)')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
