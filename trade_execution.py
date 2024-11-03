# trade_execution.py

async def place_order(symbol, side, qty, order_type="Market"):
    """
    Bybit에서 주문을 실행합니다.
    
    Parameters:
        symbol (str): 거래할 심볼 (예: "BTCUSDT")
        side (str): 주문 방향 ("Buy" 또는 "Sell")
        qty (float): 주문할 수량 (USDT 기준, 레버리지 포함)
        order_type (str): 주문 유형 (기본값: "Market")
    
    Returns:
        dict: 주문 응답
    """
    if not bybit_client:
        logging.error("Bybit 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        # 주문 실행 (레버리지는 이미 포지션 크기에 포함됨)
        response = await asyncio.to_thread(
            bybit_client.order.create,  # 수정된 메서드 이름
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            time_in_force="GoodTillCancel"
        )
        logging.info(f"{side} 주문이 실행되었습니다: {response}")
        return response
    except Exception as e:
        logging.error(f"주문 실행 중 에러 발생: {e}")
        return None
