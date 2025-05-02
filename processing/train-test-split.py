import json
import random
import pathlib
import sys

random.seed(42)                       # so you can reproduce the split

src = pathlib.Path("fine_tune_data.jsonl")
train_out = pathlib.Path("train.jsonl")
eval_out = pathlib.Path("eval.jsonl")

with src.open() as f:
    records = [json.loads(line) for line in f]

random.shuffle(records)
cut = int(len(records) * 0.9)
train, eval = records[:cut], records[cut:]

for path, split in [(train_out, train), (eval_out, eval)]:
    with path.open("w") as f:
        for r in split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"🌟 Wrote {len(train)} records to {train_out}  "
      f"and {len(eval)} to {eval_out}")
