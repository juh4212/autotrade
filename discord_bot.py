# discord_bot.py

import discord
import logging
import os
import asyncio

# Discord Intents 설정
intents = discord.Intents.default()
intents.guilds = True        # 서버 관련 이벤트 수신
intents.messages = True      # 메시지 관련 이벤트 수신

# Discord 클라이언트 초기화
client = discord.Client(intents=intents)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 환경 변수에서 Discord 채널 ID 가져오기
try:
    DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID'))
except (TypeError, ValueError):
    logging.error('DISCORD_CHANNEL_ID가 올바른 숫자가 아닙니다.')
    DISCORD_CHANNEL_ID = None

@client.event
async def on_ready():
    logging.info(f'연결되었습니다! (사용자: {client.user})')
    if DISCORD_CHANNEL_ID:
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
    프로그램 시작 시 잔고 정보를 Discord 채널에 전송합니다.
    실제 잔고 정보를 가져오는 로직으로 대체하세요.
    """
    balance_info = "현재 잔고: 10 BTC, 5000 USDT"  # 예시 메시지
    await send_message(f"프로그램이 시작되었습니다. 잔고 정보:\n{balance_info}")

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

async def run_bot():
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

def run_discord_bot():
    asyncio.run(run_bot())

if __name__ == "__main__":
    run_discord_bot()
