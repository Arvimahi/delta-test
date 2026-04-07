from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent / ".env")
pk = os.getenv("POLY_PRIVATE_KEY", "")
print(f"Key length: {len(pk)}")
print(f"Starts with 0x: {pk.startswith('0x')}")
print(f"First 6 chars: {pk[:6]}...")
print(f"Last 4 chars: ...{pk[-4:]}")

# Check if it's a valid hex string
try:
    if pk.startswith("0x"):
        int(pk[2:], 16)
    else:
        int(pk, 16)
    print("Valid hex: YES")
except:
    print("Valid hex: NO - this is the problem!")
