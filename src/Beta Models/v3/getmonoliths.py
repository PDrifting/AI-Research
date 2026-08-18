import os
import re

# Directory containing universe_map_thread_XXXX.txt files
INPUT_DIR = "./"   # change if needed

# Regex to capture MONOLITH lines
monolith_re = re.compile(
    r"MONOLITH\s+id=(\d+)\s+gate=([A-Z]+)\s+delays=\((\d+),\s*(\d+)\)"
)

results = []

for filename in os.listdir(INPUT_DIR):
    if not filename.startswith("universe_map_thread_"):
        continue
    if not filename.endswith(".txt"):
        continue

    # Extract thread ID from filename
    m = re.search(r"universe_map_thread_(\d+)\.txt", filename)
    if not m:
        continue
    thread_id = int(m.group(1))

    path = os.path.join(INPUT_DIR, filename)

    with open(path, "r") as f:
        for line in f:
            match = monolith_re.search(line)
            if match:
                monolith_id = int(match.group(1))
                gate = match.group(2)
                d0 = int(match.group(3))
                d1 = int(match.group(4))

                results.append({
                    "thread": thread_id,
                    "monolith_id": monolith_id,
                    "gate": gate,
                    "delays": (d0, d1),
                })

# Write results to monoliths.txt
with open("monoliths.txt", "w") as out:
    for r in results:
        out.write(
            f"thread={r['thread']}  monolith_id={r['monolith_id']}  "
            f"gate={r['gate']}  delays={r['delays']}\n"
        )
