import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

async def main():
    key = os.getenv("DEEPGRAM_API_KEY", "")
    print(f"Key present: {bool(key)}, length: {len(key)}")
    client = AsyncDeepgramClient(api_key=key)
    try:
        async with client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate=16000,
            eot_threshold=0.2,
        ) as conn:
            print("CONNECTED to Flux OK")
            conn.on(EventType.MESSAGE, lambda m: print("msg:", getattr(m, "type", "?")))
            # listen briefly
            await asyncio.wait_for(conn.start_listening(), timeout=3)
    except asyncio.TimeoutError:
        print("Connected, no audio sent (expected) — connection works")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

asyncio.run(main())