import re

fp = r"c:\Users\Aravind\Trading\Crypto\Poly\live_dryrun.py"
with open(fp, "r") as f:
    lines = f.readlines()

# Line 699 (0-indexed 698) has 13 spaces, should be 12
if lines[698].startswith("             except"):
    lines[698] = "            except Exception as e:\n"
    print("Fixed line 699 indentation (13 -> 12 spaces)")
else:
    print(f"Line 699 looks ok: {repr(lines[698][:30])}")

with open(fp, "w") as f:
    f.writelines(lines)
print("Done")
