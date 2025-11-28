import pandas as pd
from pymongo import MongoClient

# ---------------- Connect to MongoDB ----------------
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)

db = client["climate_db"]          # Database name
collection = db["temperatures_clean"]  # Collection name

# ---------------- Check number of documents ----------------
count = collection.count_documents({})
print(f"Documents in collection: {count}")
if count == 0:
    print("Collection is empty!")
else:
    # ---------------- Load data into DataFrame ----------------
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Check first few rows
    print(df.head())

    # Optional: remove MongoDB _id field
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    # ---------------- Save to CSV ----------------
    df.to_csv("temperatures_clean.csv", index=False)
    print("Data saved to temperatures_clean.csv successfully!")
