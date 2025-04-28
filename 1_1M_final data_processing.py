import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

# --------------------Load Dataset------------------------------------
df = pd.read_excel("Revised Code and Data/dataset/ownerclan_narosu_1M_DB_250410.xlsx")
pd.set_option('display.max_colwidth', None)
full_df = df

# Select Relevant Column: 원본상품명 (Product Name), 키워드 (Keywords), 모델명 (Model Name)
df = df[['원본상품명', '키워드', '모델명']] 

# Lowercase (important for multilingual compatibility)
def generate_processed_text(row):
    return (
        f"passage: "
        f"[product title] {str(row['원본상품명']).strip()} "
        f"[keywords] {str(row['키워드']).strip()} "
        f"[model name] {str(row['모델명']).strip()}"
    ).lower()

# -------------------- Apply Preprocessing --------------------
df["product_description"] = df.apply(generate_processed_text, axis=1)

# Clean extra whitespace
df["product_description"] = df["product_description"].str.replace(r"\s+", " ", regex=True).str.strip()

# Clean up excessive commas
df["product_description"] = df["product_description"].str.replace(r",+", ",", regex=True)

# Keep Korean, English, numbers, basic punctuation
df["product_description"] = df["product_description"].apply(
    lambda x: re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s,./:()\[\]<>-]", "", x)
)

# df.info()
# print(df["product_description"].head(2))

# 2. Merge `product_description` into it (ensure same index!)
full_df["product_description"] = df["product_description"]

full_df.to_excel("Revised Code and Data/dataset/processed_ownerclan_narosu_1M_DB_250410.xlsx", index=False)
full_df.info()

# print(full_df.head(2))


