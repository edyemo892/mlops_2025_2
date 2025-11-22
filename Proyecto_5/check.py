from steps.ingest import Ingestion
from steps.clean import Cleaner

def debug_cleaner_outputs():
    # 1. Ingesta: cargar TODO el dataset
    ingestion = Ingestion()
    data = ingestion.load_data()
    print("=== RAW DATA ===")
    print("Shape data      :", data.shape)
    print("Columnas (raw)  :", list(data.columns))
    print()

    # 2. Limpieza completa
    cleaner = Cleaner()
    data_clean = cleaner.clean_data(data)
    print("=== DATA CLEAN ===")
    print("Shape data_clean:", data_clean.shape)
    print("Columnas (clean):", list(data_clean.columns))
    print()

    # 3. Split 80/20 después de limpiar
    train_data, test_data = cleaner.split_data(data_clean)
    print("=== TRAIN / TEST (después de limpiar) ===")
    print("Shape train_data:", train_data.shape)
    print("Shape test_data :", test_data.shape)
    print("¿Mismas columnas? ->",
          list(train_data.columns) == list(test_data.columns))
    print()

    # 4. Ver primeros registros
    print(">>> Head TRAIN:")
    print(train_data.head())
    print("\n>>> Head TEST:")
    print(test_data.head())

    # 4. Ver primeros registros
    print(">>> Head TRAIN:")
    print(list(train_data.columns))
    print("\n>>> Head TEST:")
    print(list(test_data.columns))

    # 5. Info rápida de las variables numéricas
    print("\n=== DESCRIBE TRAIN (numéricas) ===")
    print(train_data.describe())

    print("\n=== DESCRIBE TEST (numéricas) ===")
    print(test_data.describe())

if __name__ == "__main__":
    debug_cleaner_outputs()
