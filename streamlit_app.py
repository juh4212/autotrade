import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
import traceback

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MongoDB 연결 함수
def get_mongo_connection():
    try:
        logger.info("MongoDB 연결 시도 중...")
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            error_msg = "MongoDB URI가 설정되지 않았습니다. .env 파일을 확인하세요."
            logger.error(error_msg)
            st.error(error_msg)
            st.stop()
        client = MongoClient(mongo_uri)
        logger.info("MongoDB에 성공적으로 연결되었습니다.")
        return client
    except Exception as e:
        error_msg = f"MongoDB 연결 오류: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

# 데이터 로드 함수
def load_data():
    try:
        logger.info("데이터 로드 시작...")
        client = get_mongo_connection()
        db = client['bitcoin_trades_db']  # 데이터베이스 이름
        trades_collection = db['trades']  # 컬렉션 이름

        # 모든 트레이드 데이터를 가져오고, timestamp 기준으로 정렬
        cursor = trades_collection.find().sort("timestamp", -1)
        trades = list(cursor)
        logger.info(f"트레이드 데이터 {len(trades)}건 로드됨.")

        if not trades:
            warning_msg = "데이터가 없습니다."
            logger.warning(warning_msg)
            st.warning(warning_msg)
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # 필요한 필드가 있는지 확인
        expected_columns = ['timestamp', 'decision', 'percentage', 'reason', 'btc_balance', 
                            'usdt_balance', 'btc_avg_buy_price', 'btc_usdt_price', 'reflection']
        for col in expected_columns:
            if col not in df.columns:
                logger.warning(f"필드 '{col}'이 데이터에 존재하지 않습니다. None으로 채웁니다.")
                df[col] = None  # 없는 필드는 None으로 채움

        # timestamp 형식 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info("데이터 로드 및 전처리 완료.")
        return df
    except Exception as e:
        error_msg = f"데이터 로드 오류: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        st.error("데이터 로드 중 오류가 발생했습니다. 로그를 확인하세요.")
        st.stop()

# 메인 함수
def main():
    try:
        st.title('Bitcoin Trades Viewer (Bybit Version)')
        logger.info("Streamlit 애플리케이션 시작.")

        # 데이터 로드
        df = load_data()

        if df.empty:
            st.stop()

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
        if 'reflection' in df.columns and not df['reflection'].isnull().all():
            recent_reflection = df[['timestamp', 'reflection']].dropna().iloc[0]['reflection']
            st.write(recent_reflection)
            logger.info("최근 반성 내용 표시 완료.")
        else:
            st.write("No reflection data available.")
            logger.info("반성 데이터가 없습니다.")

    except Exception as e:
        error_msg = f"애플리케이션 오류: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        st.error("애플리케이션 실행 중 오류가 발생했습니다. 로그를 확인하세요.")

if __name__ == "__main__":
    main()
