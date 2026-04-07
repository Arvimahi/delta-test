import os
from pathlib import Path

def fix_syntax(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        src = f.read()

    # Fix the \n literal
    src = src.replace('\\n    async def run', '    async def run')
    
    # Fix the asyncio setup
    bad_gather = "self.stream_coinbase(),`n            self.stream_binance(),"
    good_gather = "self.stream_coinbase(),\\n            self.stream_binance(),"
    src = src.replace(bad_gather, "self.stream_coinbase(),\\n            self.stream_binance(),")
    # Actually, let's just use a clean string
    
    import re
    src = re.sub(r'asyncio\.gather\(.*?\)', '''asyncio.gather(
            self.stream_coinbase(),
            self.stream_binance(),
            self._ws_broadcast_loop(),
            self.tg.poll_commands(),
        )''', src, flags=re.DOTALL)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(src)
    print(f"Fixed {filename}")

fix_syntax('live_dryrun.py')
fix_syntax('live_money.py')
