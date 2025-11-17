import os
import yaml
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model'].get('params', {})
        self.model_path = self.config['model']['store_path']
        # Semilla para replicar el split (se usa fuera de esta clase)
        self.random_state = self.config.get('training', {}).get('random_state', 104)

        self.pipeline = self.create_pipeline()
        

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
        # Ajusta las columnas según tu dataset
        preprocessor = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), 
             ['employment_type', 'job_title'])
        ], remainder='passthrough')

        # Seleccionar estimador según config (aquí LinearRegression)
        if self.model_name.lower().startswith("linear"):
            try:
                model = LinearRegression(**self.model_params)
            except Exception:
                model = LinearRegression()
        else:
            # fallback: LinearRegression si no se reconoce
            model = LinearRegression()

        steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]

        pipeline = Pipeline(steps)
        return pipeline

    def feature_target_separator(self, data):
        if 'salary_in_usd' not in data.columns:
            raise KeyError("Columna objetivo 'salary_in_usd' no encontrada en los datos")
        X = data.drop(columns='salary_in_usd')
        y = data['salary_in_usd']
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)
