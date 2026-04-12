import httpx, asyncio
async def check():
    async with httpx.AsyncClient(timeout=5) as c:
        for port, name in [(8001,"solver"),(8000,"arena"),(9009,"evaluator")]:
            try:
                r = await c.get(f"http://127.0.0.1:{port}/.well-known/agent.json")
                print(f"{name}:{port} -> {r.status_code} {r.json()['name']}")
            except Exception as e:
                print(f"{name}:{port} -> ERROR: {e}")
asyncio.run(check())
