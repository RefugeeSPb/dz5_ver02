
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#Используем yaml как словарь
def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()["train"]

    #Читаем данные
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    #Разделяем на X и y
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    #создаем модель
    model = LogisticRegression(
        max_iter=params["max_iter"],
        random_state=params["random_state"]
    )

    #название эксперимента в mlflow
    mlflow.set_experiment("hw5-iris")

    #запускаем эксперимент
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        #сохраняем модель и метрики
        Path("models").mkdir(exist_ok=True)
        Path("metrics").mkdir(exist_ok=True)

        model_path = "models/model.joblib"
        metrics_path = "metrics/metrics.json"

        joblib.dump(model, model_path)
        metrics = {"accuracy": acc, "f1_macro": f1}
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        #логируем метрики
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)

if __name__ == "__main__":
    main()
