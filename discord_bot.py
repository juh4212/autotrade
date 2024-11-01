# discord_bot.py

import discord
from discord.ext import commands
import asyncio
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID'))  # 알림을 보낼 채널 ID

# Discord 클라이언트 설정
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

async def send_discord_message(message):
    """
    지정된 Discord 채널로 메시지를 전송합니다.
    """
    try:
        channel = bot.get_channel(DISCORD_CHANNEL_ID)
        if channel:
            await channel.send(message)
        else:
            print(f"채널을 찾을 수 없습니다: {DISCORD_CHANNEL_ID}")
    except Exception as e:
        print(f"Discord 메시지 전송 에러: {e}")

def run_discord_bot():
    """
    Discord 봇을 실행합니다.
    """
    bot.run(DISCORD_TOKEN)

# notify_discord 함수은 다른 모듈에서 호출할 수 있도록 합니다.
def notify_discord(message):
    asyncio.run_coroutine_threadsafe(send_discord_message(message), bot.loop)

# 테스트용 호출
if __name__ == "__main__":
    import threading
    import time

    # Discord 봇을 별도의 스레드에서 실행
    discord_thread = threading.Thread(target=run_discord_bot, daemon=True)
    discord_thread.start()

    # 봇이 준비될 시간을 대기
    time.sleep(5)

    # 테스트 메시지 전송
    notify_discord("자동매매 봇이 시작되었습니다!")
