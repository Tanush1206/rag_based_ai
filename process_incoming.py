import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import joblib
import json


def create_embeddings(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list}
    )

    response = r.json()

    if "embeddings" not in response:
        raise Exception(response)

    return response["embeddings"]


def inference(prompt, model):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )

    response = r.json()
    return response


# Load the precomputed embeddings
df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embeddings([incoming_query])[0]
# print(f"Question Embedding: {question_embedding}")


# Find similarities of question_embedding with other embeddings
similarities = cosine_similarity(
    np.stack(df["embedding"]), [question_embedding]
).flatten()
# print(similarities)
top_result = 5
max_indx = similarities.argsort()[::-1][0:top_result]

print(max_indx)

new_df = df.loc[max_indx]
# print(new_df[["title", "number" ,"text"]])

prompt = f"""
I am teaching web development using the Sigma Web Development Course.

Below are the most relevant subtitle chunks retrieved using semantic search.

{new_df[['title','number','start','end','text']].to_json(orient='records', indent=2)}

User Question:
{incoming_query}

Instructions:
- Answer ONLY using the provided chunks.
- Mention the video number, video title and timestamps.
- If multiple timestamps are relevant, include all of them.
- If the answer is not present in the retrieved chunks, say that you couldn't find enough information.
- If the question is unrelated to the course, politely say that you can only answer questions related to the course.
"""

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt, "deepseek-r1:latest")
print(response["response"])

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response["response"])

# for index , item in new_df.iterrows():
#     print(index , item["title"], item["text"], item["number"], item["start"] , item["end"])
