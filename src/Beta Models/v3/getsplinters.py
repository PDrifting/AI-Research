import os
import re

INPUT_DIR = "./"   # adjust if needed

# Regex to capture the header line containing gate + delays
header_re = re.compile(
    r"gate=([A-Z]+)\s+delays=\((\d+),\s*(\d+)\)"
)

# Regex to capture loop_len lines
loop_re = re.compile(
    r"loop_len=(\d+)"
)

results = []

for filename in os.listdir(INPUT_DIR):
    if not filename.startswith("universe_map_thread_"):
        continue
    if not filename.endswith(".txt"):
        continue

    # Extract thread ID
    m = re.search(r"universe_map_thread_(\d+)\.txt", filename)
    if not m:
        continue
    thread_id = int(m.group(1))

    path = os.path.join(INPUT_DIR, filename)

    with open(path, "r") as f:
        lines = f.readlines()

    if not lines:
        continue

    # Check last line for DEEP HARMONIOUS - ABORTED
    if "DEEP HARMONIOUS - ABORTED" not in lines[-1]:
        continue

    # Extract gate + delays from the first matching header line
    gate = None
    d0 = None
    d1 = None

    for line in lines:
        h = header_re.search(line)
        if h:
            gate = h.group(1)
            d0 = int(h.group(2))
            d1 = int(h.group(3))
            break

    # Extract all loop_len values
    loop_lengths = []
    for line in lines:
        lm = loop_re.search(line)
        if lm:
            loop_lengths.append(int(lm.group(1)))

    if not loop_lengths:
        continue

    min_loop = min(loop_lengths)
    max_loop = max(loop_lengths)

    results.append({
        "thread": thread_id,
        "gate": gate,
        "delays": (d0, d1),
        "min_loop": min_loop,
        "max_loop": max_loop
    })

# Write splintered.txt
with open("splintered.txt", "w") as out:
    for r in results:
        out.write(
            f"thread={r['thread']}  gate={r['gate']}  "
            f"delays={r['delays']}  min_loop={r['min_loop']}  "
            f"max_loop={r['max_loop']}\n"
        )
