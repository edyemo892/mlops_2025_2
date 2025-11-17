import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score

class Predictor:
    def __init__(self):
        self.model_path = self.load_config()['model']['store_path']
        self.pipeline = self.load_model()

    def load_config(self):
        import yaml
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        X = data.drop(columns = 'salary_in_usd')
        y = data['salary_in_usd']
        return X, y

    def evaluate_model(self, X_test, y_test):
        # Predicciones
        y_pred = self.pipeline.predict(X_test)

        # Métricas de regresión
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Coeficientes: si el pipeline termina en un modelo lineal
        coef_ = None
        last_step = self.pipeline
        # Si es un Pipeline, tomamos el último step
        try:
            if hasattr(self.pipeline, "named_steps"):
                last_step = list(self.pipeline.named_steps.values())[-1]
            if hasattr(last_step, "coef_"):
                coef_ = last_step.coef_
        except Exception:
            coef_ = None

        return y_pred, coef_, mse, r2
