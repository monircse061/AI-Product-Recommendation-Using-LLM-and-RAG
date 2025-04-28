import pandas as pd
import pickle

# -------------------- Step 1: Load files --------------------
excel_path = "Revised Code and Data/processed_ownerclan_narosu_testDB_250102.xlsx"
embedding_path = "Revised Code and Data/product_description_embeddings_only_1.pkl"

# Load the full Excel data
df = pd.read_excel(excel_path)

# Load the list of embedding vectors
embedding_df = pd.read_pickle(embedding_path)

# Handle Series vs single-column DataFrame
if isinstance(embedding_df, pd.DataFrame):
    embeddings = embedding_df.iloc[:, 0].tolist()
elif isinstance(embedding_df, pd.Series):
    embeddings = embedding_df.tolist()
else:
    raise TypeError("Unexpected embedding type. Must be Series or 1-column DataFrame.")

# Ensure row count matches
assert len(df) == len(embeddings), "Row count mismatch between Excel and embedding file!"

print("Files loaded.")
print("Excel rows:", len(df))
print("Embedding vectors:", len(embeddings))

# -------------------- Step 2: Extract 8 metadata fields --------------------
metadata_fields = [
    "상품코드", "원본상품명", "오너클랜판매가", "배송비",
    "원산지", "이미지대", "이미지중", "이미지소"
]

metadata_df = df[metadata_fields]

# -------------------- Step 3: Combine embedding + metadata --------------------
combined = []

for idx in range(len(metadata_df)):
    metadata = metadata_df.iloc[idx].to_dict()
    vector = embeddings[idx]
    combined.append({
        "vector": vector,
        "metadata": metadata
    })

# -------------------- Step 4: Save combined result --------------------
output_path = "Revised Code and Data/ownerclan_vector1_and_metadata8_combined.pkl"

with open(output_path, "wb") as f:
    pickle.dump(combined, f)

print(f"Saved {len(combined)} vector+metadata records to '{output_path}'")
