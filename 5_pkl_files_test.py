import pandas as pd

# Load the pickle file
data = pd.read_pickle("embedded_chunks/chunk_350.pkl")

# Check the type of loaded object
print(f"Loaded object type: {type(data)}")

# If it's a list, display info about the first item
if isinstance(data, list):
    print(f"\nLength of list: {len(data)}")
    print(f"\nType of first item: {type(data[0])}")
    print(f"\nFirst item preview:\n{data[0]}")
    
# If it's a DataFrame, continue as expected
elif isinstance(data, pd.DataFrame):
    print(f"Total Columns: {data.shape[1]}")
    print("\nColumn Names:")
    print(data.columns.tolist())
    print("\nSample Rows:")
    pd.set_option('display.max_colwidth', None)
    print(data.head(3))
else:
    print("Unexpected data format inside the .pkl file.")



#--------------------------------------------------------------------------------------------------
# import pandas as pd

# # Load the file
# df = pd.read_pickle("Revised Code and Data/embedding data/product_embeddings.pkl")
# pd.set_option('display.max_colwidth', None)

# # See column names and shape
# print("Columns:", df.columns)
# print("Shape:", df.shape)

# # (Optional) Preview a few rows
# print(df.head(1))

# # Split the embedding column
# product_description_embedding_column = df[["embedding"]]  # keep as DataFrame
# features_df = df.drop(columns=["embedding"])  # rest of the data

# # Save separately
# product_description_embedding_column.to_pickle("Revised Code and Data/product_description_embeddings_only.pkl")
# features_df.to_pickle("Revised Code and Data/product_features_with_product_description_feature.pkl")

#--------------------------------------------------------------------------------------------------

# import pandas as pd
# import pickle

# # Load combined vector + metadata list
# with open("Revised Code and Data/embedding data/product_embeddings.pkl", "rb") as f:
#     data = pickle.load(f)

# # Confirm it's a list
# if isinstance(data, list) and isinstance(data[0], dict):
#     print("Loaded list of vector + metadata records.")
#     print("Sample record keys:", data[0].keys())
#     print("Vector length:", len(data[0]['vector']))
#     print("Metadata example:", data[0]['metadata'])
# else:
#     print("The loaded file is not in the expected format (list of dicts).")


