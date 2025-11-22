import pandas as pd
import yaml

class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        # 1. Leer ruta del dataset desde config.yml
        data_path = self.config['data']['data_path']

        # 2. Cargar el dataset completo
        data = pd.read_csv(data_path, sep=';')

        # 3. Retornar el dataset sin dividir
        return data
