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
    level=logging.INFO,  # 필요 시 DEBUG로 변경
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 환경 변수에서 Discord 채널 ID 및 Bot Token 가져오기
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

@client.event
async def on_ready():
    logging.info(f'Connected to Discord as {client.user}')

@client.event
async def on_disconnect():
    logging.warning('Disconnected from Discord')

@client.event
async def on_error(event, *args, **kwargs):
    logging.error(f'Error in event "{event}"', exc_info=True)

async def send_message(message):
    if DISCORD_CHANNEL_ID:
        channel = client.get_channel(int(DISCORD_CHANNEL_ID))
        if channel:
            try:
                await channel.send(message)
                logging.info(f'Message sent to Discord: {message}')
            except Exception as e:
                logging.error(f'Failed to send message to Discord: {e}')
        else:
            logging.error(f'Channel ID {DISCORD_CHANNEL_ID} not found.')
    else:
        logging.error('DISCORD_CHANNEL_ID is not set.')

async def post_reflection(reflection):
    """
    AI의 반성 내용을 Discord 채널에 게시하는 함수
    """
    if reflection:
        message = f"📊 **AI Trading Reflection:**\n{reflection}"
        await send_message(message)
    else:
        logging.error("Reflection content is empty.")

async def run_discord_bot():
    if not DISCORD_BOT_TOKEN:
        logging.error('DISCORD_BOT_TOKEN is not set.')
        return
    try:
        await client.start(DISCORD_BOT_TOKEN)
    except Exception as e:
        logging.error(f'Error starting Discord bot: {e}')

def run_discord_bot_task(loop):
    asyncio.run_coroutine_threadsafe(run_discord_bot(), loop)
