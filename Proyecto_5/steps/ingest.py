#from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        #train_data_path = self.config['data']['train_path']
        #test_data_path = self.config['data']['test_path']
        data_path=self.config['data']['data_path'] 
        #train_data = pd.read_csv(train_data_path)
        #test_data = pd.read_csv(test_data_path)
        data = pd.read_csv(data_path)
        #En caso si quiesiera hacer con sklearn
        #train_data, test_data = train_test_split(
        #     df,
        #     test_size=0.2,
        #     random_state=42
        #) 

        #Como tengo pandas, lo hago de esta forma
        split = int(len(data) * 0.8)
        train_data = data.iloc[:split]     # primeras filas
        test_data  = data.iloc[split:]     # últimas filas
        return train_data, test_data
