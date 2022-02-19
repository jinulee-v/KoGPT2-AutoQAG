import argparse
import json
import pandas as pd
import os

folder = "raw_data"
files = os.listdir(os.path.join(os.getcwd(), folder))
files = [os.path.join(os.getcwd(), folder, name) for name in files if name.endswith('.json')]

result = []
for file in files:
  with open(file, "r") as f:
    data = json.load(f)
    print(json.dumps(data["data"][0]["paragraphs"][0], indent=4, ensure_ascii=False))

    for doc in data["data"]:
      for para in doc["paragraphs"]:
        text = [para["context"].strip().replace("\n", " ")]
        for qa in para["qas"]:
          text += [qa["question"].strip().replace("\n", " "), qa["answers"][0]["text"].strip().replace("\n", " ")]
        result.append("\t".join(text))
with open("raw_data/train.txt", "w", encoding="UTF-8") as file:
  file.write("\n".join(result))