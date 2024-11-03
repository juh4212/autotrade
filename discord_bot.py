# discord_bot.py

import asyncio
from data_collection import get_wallet_balance  # 잔고 정보를 가져오는 함수 임포트

@client.event
async def on_ready():
    logging.info(f'연결되었습니다! (사용자: {client.user})')
    await send_message("📢 **봇이 연결되었습니다!**")
    
    # 잔고 정보 가져오기
    if bybit_client:
        balance_info = await asyncio.to_thread(get_wallet_balance)  # 비동기적으로 잔고 정보 가져오기
        if balance_info:
            equity = balance_info.get("equity", 0)
            available_to_withdraw = balance_info.get("available_to_withdraw", 0)
            balance_message = (
                f"💰 **현재 잔고 정보:**\n"
                f"**Equity:** {equity} USDT\n"
                f"**Available to Withdraw:** {available_to_withdraw} USDT"
            )
            await send_message(balance_message)
        else:
            await send_message("❌ **잔고 정보를 가져오지 못했습니다.**")
    else:
        await send_message("❌ **Bybit 클라이언트가 초기화되지 않았습니다.**")
