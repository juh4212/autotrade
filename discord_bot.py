# discord_bot.py

import discord
import logging
import os
import asyncio

# Discord Intents 설정
intents = discord.Intents.default()
intents.messages = True  # 필요에 따라 활성화

# Discord 클라이언트 초기화
client = discord.Client(intents=intents)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 수준 설정
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()  # 콘솔에 로그 출력
    ]
)

# 환경 변수에서 Discord 채널 ID 가져오기
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', 'YOUR_CHANNEL_ID_HERE'))  # .env 파일에 설정 추천

@client.event
async def on_ready():
    logging.info('연결되었습니다! (사용자: {0.user})')
    await send_message('연결되었습니다!')

@client.event
async def on_disconnect():
    logging.warning('연결이 끊겼습니다!')
    # 직접적으로 on_disconnect에서는 비동기 함수를 호출할 수 없으므로, 별도의 스레드에서 호출
    asyncio.create_task(send_message('연결이 끊겼습니다!'))

@client.event
async def on_error(event, *args, **kwargs):
    logging.error(f'이벤트 "{event}" 처리 중 에러 발생', exc_info=True)

async def send_message(message):
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        await channel.send(message)
    else:
        logging.error(f'채널 ID {DISCORD_CHANNEL_ID}을 찾을 수 없습니다.')

def notify_discord(message):
    # 비동기 이벤트 루프에서 안전하게 메시지를 전송
    asyncio.run_coroutine_threadsafe(send_message(message), client.loop)

async def run_bot():
    token = os.getenv('DISCORD_BOT_TOKEN')  # .env 파일이나 환경 변수로 관리하는 것을 권장
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
