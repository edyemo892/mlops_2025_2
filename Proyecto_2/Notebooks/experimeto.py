import mlflow

# Opcional: aseguramos que los datos se guarden en ./mlruns
mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.85)
    print("Ejecución registrada en MLflow.")
