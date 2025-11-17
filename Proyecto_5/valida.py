# ...existing code...
import logging
from steps.ingest import Ingestion
from steps.clean import Cleaner
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def inspect_df(df: pd.DataFrame, name: str):
    logging.info(f"{name} shape: {df.shape}")
    logging.info(f"{name} dtypes:\n{df.dtypes}")
    logging.info(f"{name} nulls:\n{df.isnull().sum()}")
    logging.info(f"{name} head:\n{df.head().to_string(index=False)}")
    logging.info(f"{name} describe:\n{df.describe(include='all').to_string()}")

def main():
    ing = Ingestion()
    train, test = ing.load_data()
    inspect_df(train, "train (raw)")
    inspect_df(test, "test (raw)")

    cleaner = Cleaner()
    train_clean = cleaner.clean_data(train)
    test_clean = cleaner.clean_data(test)
    inspect_df(train_clean, "train (clean)")
    inspect_df(test_clean, "test (clean)")

    # opcional: guardar para inspección externa
    train_clean.to_csv("train_clean.csv", index=False)
    test_clean.to_csv("test_clean.csv", index=False)
    logging.info("Guardados: train_clean.csv y test_clean.csv")

if __name__ == "__main__":
    main()
# ...existing code...