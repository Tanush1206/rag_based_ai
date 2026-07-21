import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


def create_embeddings(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list}
    )

    response = r.json()

    if "embeddings" not in response:
        raise Exception(response)

    return response["embeddings"]


jsons = sorted(os.listdir("jsons"))

my_dicts = []
chunk_id = 0

BATCH_SIZE = 200

for json_file in jsons:
    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    texts = [chunk["text"] for chunk in content["chunks"]]

    embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        embeddings.extend(create_embeddings(batch))

    if len(embeddings) != len(content["chunks"]):
        raise Exception(
            f"Expected {len(content['chunks'])} embeddings, got {len(embeddings)}"
        )

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
df = pd.DataFrame.from_records(my_dicts)
#save this DataFrame using joblib
joblib.dump(df , "embeddings.joblib")
print(df)

print(df.shape)
print(df.columns)
