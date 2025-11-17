import logging
import yaml
# import mlflow
# import mlflow.sklearn
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    # Clean data
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    #Prepare and train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # Evaluate model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    y_pred, coef_, mse, r2= predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    #Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")

    # formatear coeficientes de forma segura
    if coef_ is None:
        print("Coefficients: None")
    else:
        import numpy as np
        coef_arr = np.asarray(coef_).flatten()
        if coef_arr.size == 0:
            print("Coefficients: []")
        else:
            # mostrar hasta 10 coeficientes con 4 decimales
            display_count = min(10, coef_arr.size)
            coef_str = ", ".join(f"{c:.4f}" for c in coef_arr[:display_count])
            if coef_arr.size > display_count:
                coef_str += f", ... (total {coef_arr.size})"
            print(f"Coefficients: {coef_str}")

    print(f"Mean squared error: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    print("=====================================================\n")


def train_with_mlflow():
    """
    Ejecuta el flujo completo bajo un run de MLflow.
    Adaptado para regresión: registra MSE y R2, parámetros y guarda el pipeline.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except Exception as e:
        raise RuntimeError("mlflow no está instalado en el entorno. Instálalo con: python -m pip install mlflow") from e

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    exp_name = config.get('mlflow', {}).get('experiment_name', "Model Training Experiment")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Clean data
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Prepare and train model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")

        # Evaluate model
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        # predictor.evaluate_model devuelve: y_pred, coef_, mse, r2
        y_pred, coef_, mse, r2 = predictor.evaluate_model(X_test, y_test)
        logging.info("Model evaluation completed successfully")

        # Log tags, params y métricas
        mlflow.set_tag('Model developer', 'prsdm')
        mlflow.set_tag('preprocessing', 'OneHotEncoder + passthrough')

        model_params = config.get('model', {}).get('params', {})
        mlflow.log_params(model_params)
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("r2", float(r2))

        # Log model artifact
        try:
            mlflow.sklearn.log_model(trainer.pipeline, "model")
            # Opcional registro en el Model Registry (si hay servidor)
            model_name = config.get('mlflow', {}).get('registered_name', "insurance_model")
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.register_model(model_uri, model_name)
            except Exception:
                # puede fallar si no hay Model Registry disponible; no bloquea
                logging.info("Registro de modelo en MLflow falló o no está disponible; se ignoró.")
        except Exception as e:
            logging.warning(f"No se pudo guardar el modelo en MLflow: {e}")

        logging.info("MLflow tracking completed successfully")

        # Imprimir resultados completos (mismo formato que main)
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")

        if coef_ is None:
            print("Coefficients: None")
        else:
            import numpy as np
            coef_arr = np.asarray(coef_).flatten()
            if coef_arr.size == 0:
                print("Coefficients: []")
            else:
                display_count = min(10, coef_arr.size)
                coef_str = ", ".join(f"{c:.4f}" for c in coef_arr[:display_count])
                if coef_arr.size > display_count:
                    coef_str += f", ... (total {coef_arr.size})"
                print(f"Coefficients: {coef_str}")

        print(f"Mean squared error: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print("=====================================================\n")


if __name__ == "__main__":
     main()
     train_with_mlflow()
