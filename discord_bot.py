# discord_bot.py

import discord
import logging
import os
import asyncio
from pybit.unified_trading import HTTP

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
    await list_channels()  # 모든 채널 목록 출력

@client.event
async def on_disconnect():
    logging.warning('연결이 끊겼습니다!')
    if DISCORD_CHANNEL_ID:
        asyncio.create_task(send_message('연결이 끊겼습니다!'))

@client.event
async def on_error(event, *args, **kwargs):
    logging.error(f'이벤트 "{event}" 처리 중 에러 발생', exc_info=True)

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
    try:
        await client.start(token)
    except Exception as e:
        logging.error(f'Discord 봇 실행 중 에러 발생: {e}')

def run_discord_bot_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_discord_bot())
