import os
import time
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# -------------------- Configuration --------------------
OPENAI_API_KEY = ""
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_DIM = 1536
CHUNK_SIZE = 1000
BATCH_SIZE = 100
CHECKPOINT_DIR = "embedded_chunks"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------- Initialize OpenAI Client --------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Load Data --------------------
df = pd.read_excel("Revised Code and Data/dataset/processed_ownerclan_narosu_1M_DB_250410.xlsx")
print(f"Total samples: {len(df)}")

# -------------------- Split into Chunks --------------------
chunks = [df[i:i + CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]

# -------------------- Retry Embedding Function --------------------
def get_embedding_with_retry(client, text, model, max_retries=2):
    for attempt in range(max_retries):
        try:
            text = text.replace("\n", " ")  # remove line breaks
            response = client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as e:
            wait = 2 ** attempt + random.random()
            print(f"Retry {attempt + 1}: {e}. Waiting {wait:.2f}s")
            time.sleep(wait)
    return None

# -------------------- Create Batches --------------------
def create_batches(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df[i:i + batch_size]

# -------------------- Main Loop --------------------
start_time = time.time()
failed_log_path = "failed_embeddings.txt"
if os.path.exists(failed_log_path):
    os.remove(failed_log_path)

for chunk_index, chunk_df in enumerate(chunks):
    save_path = f"{CHECKPOINT_DIR}/chunk_{chunk_index}.pkl"
    if os.path.exists(save_path):
        print(f"Skipping chunk {chunk_index}: already exists.")
        continue

    print(f"Processing chunk {chunk_index + 1}/{len(chunks)}")
    embeddings = []

    for batch in tqdm(create_batches(chunk_df, BATCH_SIZE), desc=f"🔄 Embedding Chunk {chunk_index}"):
        for local_id, row in batch.iterrows():
            text = row["product_description"]
            if pd.isna(text) or text.strip() == "":
                embedding = [0.0] * VECTOR_DIM
            else:
                embedding = get_embedding_with_retry(client, text, EMBEDDING_MODEL)
                if embedding is None:
                    embedding = [0.0] * VECTOR_DIM
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{row.get('상품코드', 'unknown')},{chunk_index}_{local_id}\n")
            embeddings.append(embedding)
        time.sleep(0.5)  # Sleep after each batch to avoid hitting OpenAI rate limits

    chunk_df["embedding"] = embeddings
    chunk_df.to_pickle(save_path)
    print(f"Saved: {save_path}")

end_time = time.time()
print(f"Finished in {(end_time - start_time) / 60:.2f} minutes.")
