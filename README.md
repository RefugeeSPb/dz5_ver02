# Проект: MLOps‑подход для датасета Iris (DVC + Feast)

## Описание

Проект демонстрирует базовые практики MLOps на примере задачи классификации Iris:  
версионирование данных с помощью DVC и использование Feast в роли локального feature store для подготовки признаков.  
Решение ориентировано на офлайн‑эксперименты и учебное знакомство с инфраструктурными компонентами ML‑систем.

## Стек технологий

- Python 3
- DVC — контроль версий данных и экспериментов
- Feast 0.62.0 — управление признаками (Feature Store)
- pandas, scikit‑learn — подготовка данных и базовая модель

## Структура проекта

```text
dz5_ver02/
  iris_feature_repo/
    feature_repo/
      feature_store.yaml      # конфигурация Feast
      iris_features.py        # Entity + FeatureView для Iris
      feature_definitions.py  # пример конфигурации из шаблона Feast
      data/
        iris.parquet          # датасет с признаками, label, flower_id, event_timestamp
        registry.db           # реестр Feast
        online_store.db       # локальный online-store (SQLite)
  05_01_DVC.ipynb             # ноутбук по DVC и версионированию данных
  05_02_FeatureStores.ipynb   # ноутбук по Feast и Feature Store
```

## Как запустить

1. Установить зависимости (пример):

   ```bash
   pip install -r requirements.txt
   ```

   (либо установить `feast`, `dvc`, `pandas`, `scikit-learn` вручную.)

2. Перейти в каталог с feature‑репозиторием Feast:

   ```bash
   cd dz5_ver02/iris_feature_repo/feature_repo
   ```

3. Применить конфигурацию Feast:

   ```bash
   feast apply
   ```

   Эта команда создаст/обновит реестр (`data/registry.db`) и таблицы в локальном SQLite‑хранилище [web:410].

4. Получить исторические признаки из Feast (пример в ноутбуке `05_02_FeatureStores.ipynb`):

   ```python
   import pandas as pd
   from feast import FeatureStore

   store = FeatureStore(repo_path=".")

   entity_df = pd.DataFrame(
       {
           "flower_id": ,
           "event_timestamp": [pd.Timestamp("2026-01-01")] * 3,
       }
   )

   training_df = store.get_historical_features(
       entity_df=entity_df,
       features=[
           "iris_features:sepal length (cm)",
           "iris_features:sepal width (cm)",
           "iris_features:petal length (cm)",
           "iris_features:petal width (cm)",
           "iris_features:label",
       ],
   ).to_df()

   print(training_df.head())
   ```

5. Версионирование данных и экспериментов выполняется через DVC (подробности и команды приведены в ноутбуке `05_01_DVC.ipynb`).

## Статус и выводы

- Данные приведены к единому формату (Parquet) и дополнены служебными полями (`flower_id`, `event_timestamp`), что позволяет воспроизводимо обучать модель на конкретных версиях набора данных.
- Настроен локальный Feature Store на базе Feast: определены сущности и feature views, данные доступны как для офлайн‑обучения, так и для онлайн‑хранилища (SQLite).
- Использование DVC обеспечивает версионирование данных и упрощает возврат к нужным экспериментам.
- Модель пока используется в офлайн‑режиме и не обёрнута в сервис (нет API/контейнера), а мониторинг и логирование ошибок не настроены, поэтому решение рассматривается как учебный прототип с упором на работу с данными и признаками, а не как полностью production‑ready ML‑система.
