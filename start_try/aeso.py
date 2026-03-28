import asyncio

async def fetch(name,delay):
    await asyncio.sleep(delay)
    return f"{name}完成"

async def main():
    results = await asyncio.gather(
        fetch("a",1),
        fetch("b",2),
        fetch("c",3),
    )
    print(results)

asyncio.run(main())
"""
时间(s) →   0        1        2        3
           |--------|--------|--------|

任务 a:     [ sleep 1s ]✓完成
任务 b:     [------ sleep 2s ------]✓完成
任务 c:     [----------- sleep 3s -----------]✓完成

main():     启动 gather ---------------------- 等全部完成后继续
print():                                         在 t=3s 左右打印结果
"""