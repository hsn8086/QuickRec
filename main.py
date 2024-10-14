from pathlib import Path

from quickrec.scanner.platform.wechat import get_transactions

op_dir = Path("output")
op_dir.mkdir(exist_ok=True)
with Path("img_5.png").open("rb") as f:
    for i in get_transactions(f):
        print(i)
