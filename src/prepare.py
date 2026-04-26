import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

#Используем yaml как словарь
def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()["prepare"]
    test_size = params["test_size"] #размер тестовой выборки
    random_state = params["random_state"] #рандомное число

    df = pd.read_csv("data/raw/iris.csv") #читаем данные
    X = df.drop(columns=["label"]) #готовим данные
    y = df["label"]

    #разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    ) 

    #создаем папку для сохранения
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    #сохраняем данные
    X_train.assign(label=y_train).to_csv(out_dir / "train.csv", index=False)
    X_test.assign(label=y_test).to_csv(out_dir / "test.csv", index=False)

if __name__ == "__main__":
    main()    
