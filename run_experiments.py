import argparse
import hashlib
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from clearml import Dataset
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

EXPERIMENT_NAME = "miit-mlops-practice-4-bychkova"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Запуск MLflow-экспериментов на Wine Dataset"
    )

    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow Tracking URI",
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="ClearML Dataset ID",
    )

    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="Ожидаемый SHA-256 датасета",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Каталог для загрузки датасета",
    )

    return parser.parse_args()

def build_configurations():
    return [
        {
            "name": "logreg-c-0.1",
            "family": "logistic_regression",
            "params": {
                "C": 0.1,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "logreg-c-1",
            "family": "logistic_regression",
            "params": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "logreg-c-10",
            "family": "logistic_regression",
            "params": {
                "C": 10.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "rf-depth-3",
            "family": "random_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 3,
                "min_samples_split": 2,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "rf-depth-5",
            "family": "random_forest",
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_split": 2,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "name": "rf-depth-none",
            "family": "random_forest",
            "params": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": RANDOM_STATE,
            },
        },
    ]


def build_model(config):
    params = config["params"]

    if config["family"] == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["C"],
                solver=params["solver"],
                max_iter=params["max_iter"],
                random_state=RANDOM_STATE,
            )),
        ])

    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=RANDOM_STATE,
    )


def calculate_metrics(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision_weighted": precision_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
    }

def run_all_experiments(
    experiment_id,
    configurations,
    X_train,
    y_train,
    X_val,
    y_val,
    dataset_id,
    dataset_sha256,
):
    created_run_ids = []
    trained_models = {}

    for config in configurations:
        family = config["family"]
        model = build_model(config)
        params = config["params"]

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=config["name"],
        ) as run:

            # Обучаем модель и измеряем время
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            training_seconds = time.perf_counter() - start_time

            # Рассчитываем метрики на validation
            val_metrics = calculate_metrics(
                model, X_val, y_val
            )

            # Логируем тип модели и параметры
            model_type = type(
                model.named_steps["classifier"]
                if isinstance(model, Pipeline)
                else model
            ).__name__

            mlflow.log_param("model_type", model_type)

            mlflow.log_params({
                key: "None" if value is None else value
                for key, value in params.items()
            })

            # Логируем метрики и время
            mlflow.log_metrics({
                **val_metrics,
                "training_seconds": training_seconds,
            })

            # Добавляем обязательные теги
            mlflow.set_tags({
                "model_family": family,
                "dataset_id": dataset_id,
                "dataset_sha256": dataset_sha256,
            })

            # Создаём Signature и пример входных данных
            signature = infer_signature(
                X_train,
                model.predict(X_train),
            )

            log_model_kwargs = {
                "sk_model": model,
                "name": "model",
                "signature": signature,
                "input_example": X_train.iloc[:5],
            }

            # Разрешаем конкретный тип дерева для Random Forest
            if family == "random_forest":
                log_model_kwargs["skops_trusted_types"] = [
                    "sklearn.tree._tree.Tree"
                ]

            mlflow.sklearn.log_model(**log_model_kwargs)

            created_run_ids.append(run.info.run_id)
            trained_models[run.info.run_id] = model

            print(
                f"Завершён запуск: {config['name']} | "
                f"F1 validation: {val_metrics['f1_weighted']:.4f} | "
                f"Время: {training_seconds:.4f} сек."
            )

    if len(created_run_ids) != len(configurations):
        raise RuntimeError(
            "Не все конфигурации были успешно запущены."
        )

    print("\nВсего создано запусков:", len(created_run_ids))

    return created_run_ids, trained_models


def main():
    args = parse_args()

    tracking_uri = (
        args.tracking_uri
        or __import__("os").environ.get("MLFLOW_TRACKING_URI")
        or "sqlite:///mlflow.db"
    )

    mlflow.set_tracking_uri(tracking_uri)

    print("Python:", __import__("sys").version)
    print("MLflow:", mlflow.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("Tracking URI:", tracking_uri)
    print("Dataset ID:", args.dataset_id)
    print("Expected SHA-256:", args.expected_sha256)

    # Получаем датасет из ClearML
    dataset = Dataset.get(dataset_id=args.dataset_id)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset.get_mutable_local_copy(
        target_folder=str(data_dir),
        overwrite=True,
    )

    # Проверяем наличие файла
    wine_path = data_dir / "wine.csv"

    if not wine_path.exists():
        raise FileNotFoundError(
            f"Файл wine.csv не найден: {wine_path}"
        )

    # Проверяем SHA-256
    actual_sha256 = hashlib.sha256(
        wine_path.read_bytes()
    ).hexdigest()

    print("Actual SHA-256:", actual_sha256)

    if actual_sha256 != args.expected_sha256:
        raise ValueError(
            "SHA-256 датасета не совпадает с ожидаемым!"
        )

    print("SHA-256 проверен успешно.")

    # Загружаем данные
    df = pd.read_csv(wine_path)

    X = df.drop(columns="target")
    y = df["target"]

    print("Dataset shape:", df.shape)
    print("Features:", X.shape[1])
    print("Classes:")
    print(y.value_counts().sort_index())

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    print("\nРазмеры выборок:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

        # Создаём или получаем MLflow Experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    print("\nMLflow Experiment ID:", experiment_id)

    # Запускаем все шесть экспериментов
    configurations = build_configurations()

    created_run_ids, trained_models = run_all_experiments(
        experiment_id=experiment_id,
        configurations=configurations,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        dataset_id=args.dataset_id,
        dataset_sha256=actual_sha256,
    )

    # Получаем завершённые запуски
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas",
    )

    # Оставляем только запуски текущего выполнения скрипта
    runs_df = runs_df[
        runs_df["run_id"].isin(created_run_ids)
        & (runs_df["status"] == "FINISHED")
    ].copy()

    if len(runs_df) != 6:
        raise RuntimeError(
            f"Ожидалось 6 завершённых запусков, получено: {len(runs_df)}"
        )

    # Выбираем лучшую модель по validation F1
    best_run = runs_df.sort_values(
        "metrics.f1_weighted",
        ascending=False,
    ).iloc[0]

    best_run_id = best_run["run_id"]
    best_model_uri = f"runs:/{best_run_id}/model"

    print("\nЛучшая модель:", best_run["tags.mlflow.runName"])
    print("Best Run ID:", best_run_id)
    print("Validation F1:", best_run["metrics.f1_weighted"])

    # Загружаем лучшую модель из MLflow
    original_model = trained_models[best_run_id]
    loaded_model = mlflow.sklearn.load_model(best_model_uri)

    original_predictions = original_model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)

    predictions_match = (
        original_predictions == loaded_predictions
    ).all()

    if not predictions_match:
        raise RuntimeError("Предсказания загруженной модели не совпадают!")

    # Рассчитываем тестовые метрики только после выбора модели
    test_metrics = calculate_metrics(
        loaded_model,
        X_test,
        y_test,
    )

    # Записываем тестовые метрики в лучший запуск
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics({
            f"test_{name}": value
            for name, value in test_metrics.items()
        })

    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nПредсказания совпадают:", predictions_match)
    print("Эксперименты успешно завершены.")
    print("Experiment ID:", experiment_id)
    print("Best Run ID:", best_run_id)
    print("Test F1:", test_metrics["f1_weighted"])


if __name__ == "__main__":
    main()