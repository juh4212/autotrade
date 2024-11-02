# discord_bot.py

import discord
import logging
import os
import asyncio
from pybit.unified_trading import HTTP  # pybit v5에서 HTTP 클래스 임포트

# Discord Intents 설정
intents = discord.Intents.default()
intents.guilds = True        # 서버 관련 이벤트 수신
intents.messages = True      # 메시지 관련 이벤트 수신

# Discord 클라이언트 초기화
client = discord.Client(intents=intents)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 필요 시 DEBUG로 변경
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 환경 변수에서 Discord 채널 ID 및 Bybit API 키 가져오기
try:
    DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID'))
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
except (TypeError, ValueError):
    logging.error('DISCORD_CHANNEL_ID가 올바른 숫자가 아니거나, BYBIT_API_KEY/SECRET이 설정되지 않았습니다.')
    DISCORD_CHANNEL_ID = None
    BYBIT_API_KEY = None
    BYBIT_API_SECRET = None

# Bybit 클라이언트 초기화
if BYBIT_API_KEY and BYBIT_API_SECRET:
    bybit_client = HTTP(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET
    )
    logging.info("Bybit 클라이언트가 초기화되었습니다.")
else:
    bybit_client = None
    logging.error("Bybit API 키 또는 시크릿이 설정되지 않았습니다.")

@client.event
async def on_ready():
    logging.info(f'연결되었습니다! (사용자: {client.user})')
    if DISCORD_CHANNEL_ID and bybit_client:
        await send_balance_message()
    await list_channels()  # 모든 채널 목록 출력

@client.event
async def on_disconnect():
    logging.warning('연결이 끊겼습니다!')
    if DISCORD_CHANNEL_ID:
        asyncio.create_task(send_message('연결이 끊겼습니다!'))

@client.event
async def on_error(event, *args, **kwargs):
    logging.error(f'이벤트 "{event}" 처리 중 에러 발생', exc_info=True)

async def send_balance_message():
    """
    프로그램 시작 시 실시간 잔고 정보를 Discord 채널에 전송합니다.
    Bybit API를 통해 잔고 정보를 가져옵니다.
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return

    try:
        # 잔고 정보 가져오기 (예: USDT 잔고, accountType='CONTRACT')
        response = bybit_client.get_wallet_balance(coin="USDT", accountType="CONTRACT")
        logging.debug(f"get_wallet_balance response: {response}")  # 디버그용 로그 추가

        if response.get("retCode") == 0:
            result = response.get("result")
            if result and "list" in result:
                account_list = result["list"]
                usdt_balance = None
                equity = None
                available_to_withdraw = None

                for account in account_list:
                    if account.get("accountType") == "CONTRACT":
                        coins = account.get("coin", [])
                        for coin_info in coins:
                            if coin_info.get("coin") == "USDT":
                                equity = coin_info.get("equity")
                                available_to_withdraw = coin_info.get("availableToWithdraw")
                                usdt_balance = f"Equity: {equity} USDT\nAvailable to Withdraw: {available_to_withdraw} USDT"
                                break
                    if usdt_balance:
                        break

                if usdt_balance:
                    balance_info = f"현재 잔고 정보:\n{usdt_balance}"
                    await send_message(f"프로그램이 시작되었습니다.\n{balance_info}")
                else:
                    error_msg = "USDT 잔고 정보를 찾을 수 없습니다."
                    logging.error(error_msg)
                    await send_message(error_msg)
            else:
                error_msg = "list 키가 응답에 포함되지 않았습니다."
                logging.error(error_msg)
                logging.debug(f"Full response: {response}")  # 응답 전체 로그 추가
                await send_message(error_msg)
        else:
            error_msg = f"잔고 정보 가져오기 실패: {response.get('retMsg')}"
            logging.error(error_msg)
            logging.debug(f"Full response: {response}")  # 응답 전체 로그 추가
            await send_message(error_msg)
    except KeyError as ke:
        error_msg = f"잔고 정보 파싱 중 KeyError 발생: {ke}"
        logging.error(error_msg)
        logging.debug(f"Full response: {response}")  # 응답 전체 로그 추가
        await send_message(error_msg)
    except Exception as e:
        error_msg = f"잔고 정보 가져오기 중 에러 발생: {e}"
        logging.error(error_msg)
        await send_message(error_msg)

async def send_message(message):
    if DISCORD_CHANNEL_ID:
        channel = client.get_channel(DISCORD_CHANNEL_ID)
        if channel:
            try:
                await channel.send(message)
                logging.info(f'메시지 전송: {message}')
            except Exception as e:
                logging.error(f'메시지 전송 실패: {e}')
        else:
            logging.error(f'채널 ID {DISCORD_CHANNEL_ID}을 찾을 수 없습니다.')
    else:
        logging.error('DISCORD_CHANNEL_ID가 설정되지 않았습니다.')

def notify_discord(message):
    if DISCORD_CHANNEL_ID and client.is_ready():
        asyncio.run_coroutine_threadsafe(send_message(message), client.loop)
    else:
        logging.warning('Discord 봇이 준비되지 않았거나, DISCORD_CHANNEL_ID가 설정되지 않았습니다.')

async def list_channels():
    for guild in client.guilds:
        logging.info(f'서버: {guild.name} (ID: {guild.id})')
        for channel in guild.text_channels:
            logging.info(f' - 채널: {channel.name} (ID: {channel.id})')

async def run_discord_bot():
    token = os.getenv('DISCORD_BOT_TOKEN')  # .env 파일이나 환경 변수로 관리
    if not token:
        logging.error('DISCORD_BOT_TOKEN 환경 변수가 설정되지 않았습니다.')
        return
    while True:
        try:
            await client.start(token)
        except discord.ConnectionClosed as e:
            logging.warning(f'Discord 봇 연결이 끊겼습니다: {e}')
            await asyncio.sleep(5)  # 재연결 전에 잠시 대기
        except Exception as e:
            logging.error(f'Discord 봇 실행 중 에러 발생: {e}')
            await asyncio.sleep(5)  # 재연결 전에 잠시 대기

def run_discord_bot_thread():
    asyncio.run(run_discord_bot())
