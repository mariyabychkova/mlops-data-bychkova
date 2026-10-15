
from pathlib import Path
import argparse

import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "iris":
        dataset = load_iris(as_frame=True)
    elif dataset_name == "wine":
        dataset = load_wine(as_frame=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = dataset.frame.copy()

    # Сортируем по исходным признакам и целевой переменной.
    sort_columns = list(dataset.feature_names) + ["target"]
    df = df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    return df


def prepare_base() -> None:
    """Создаёт только исходный Iris."""
    iris = load_dataset("iris")
    save_csv(iris, DATA_DIR / "iris.csv")


def prepare_wine() -> None:
    """Добавляет Wine, не изменяя существующий Iris."""
    wine = load_dataset("wine")
    save_csv(wine, DATA_DIR / "wine.csv")


def prepare_iris_features() -> None:
    """Добавляет два вычисляемых признака только в Iris."""
    iris = load_dataset("iris")

    iris["sepal_area"] = iris["sepal length (cm)"] * iris["sepal width (cm)"]
    iris["petal_area"] = iris["petal length (cm)"] * iris["petal width (cm)"]

    save_csv(iris, DATA_DIR / "iris.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible Iris and Wine datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["base", "wine", "iris_features"],
        required=True,
        help="Dataset preparation mode",
    )
    args = parser.parse_args()

    if args.mode == "base":
        prepare_base()
    elif args.mode == "wine":
        prepare_wine()
    elif args.mode == "iris_features":
        prepare_iris_features()


if __name__ == "__main__":
    main()
