import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

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

# 메인 함수
def main():
    st.title('Bitcoin Trades Viewer (Bybit Version)')

    # 데이터 로드
    df = load_data()

    # 날짜 형식 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])

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
    fig = px.pie(values=decision_counts.values, names=decision_counts.index, title='Trade Decisions')
    st.plotly_chart(fig)

    # BTC 잔액 변화
    st.header('BTC Balance Over Time')
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC Balance (BTC)')
    st.plotly_chart(fig)

    # USDT 잔액 변화
    st.header('USDT Balance Over Time')
    fig = px.line(df, x='timestamp', y='usdt_balance', title='USDT Balance')
    st.plotly_chart(fig)

    # BTC 평균 매수 가격 변화
    st.header('BTC Average Buy Price Over Time')
    fig = px.line(df, x='timestamp', y='btc_avg_buy_price', title='BTC Average Buy Price (USDT)')
    st.plotly_chart(fig)

    # BTC 현재 가격 변화
    st.header('BTC Price Over Time')
    fig = px.line(df, x='timestamp', y='btc_usdt_price', title='BTC Price (USDT)')
    st.plotly_chart(fig)

    # 투자 퍼포먼스 계산 및 표시
    st.header('Investment Performance')
    df['total_asset'] = df['usdt_balance'] + df['btc_balance'] * df['btc_usdt_price']
    fig = px.line(df, x='timestamp', y='total_asset', title='Total Asset Over Time (USDT)')
    st.plotly_chart(fig)

    # 최근 반성 및 개선 내용 표시
    st.header('Recent Reflection')
    if 'reflection' in df.columns:
        st.write(df[['timestamp', 'reflection']].dropna().tail(1)['reflection'].values[0])
    else:
        st.write("No reflection data available.")

if __name__ == "__main__":
    main()
