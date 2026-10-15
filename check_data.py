
from pathlib import Path
import hashlib
import sys

import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def expected_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "iris":
        dataset = load_iris(as_frame=True)
    elif dataset_name == "wine":
        dataset = load_wine(as_frame=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = dataset.frame.copy()
    sort_columns = list(dataset.feature_names) + ["target"]

    return (
        df.sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )


def check_dataset(dataset_name: str, expected_features: bool = False) -> None:
    path = DATA_DIR / f"{dataset_name}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Expected file is missing: {path}")

    df = pd.read_csv(path)
    original = expected_dataset(dataset_name)

    if any(column.startswith("Unnamed:") for column in df.columns):
        raise ValueError(f"{path}: unexpected index column")

    expected_rows = len(original)

    if len(df) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} rows, got {len(df)}"
        )

    if df.isna().any().any():
        raise ValueError(f"{path}: missing values found")

    if expected_features:
        required_features = ["sepal_area", "petal_area"]

        for feature in required_features:
            if feature not in df.columns:
                raise ValueError(f"{path}: missing feature {feature}")

        expected_columns = list(original.columns) + required_features

        if list(df.columns) != expected_columns:
            raise ValueError(
                f"{path}: unexpected columns or column order\n"
                f"Expected: {expected_columns}\n"
                f"Actual:   {list(df.columns)}"
            )

        expected_sepal = (
            df["sepal length (cm)"] * df["sepal width (cm)"]
        )
        expected_petal = (
            df["petal length (cm)"] * df["petal width (cm)"]
        )

        if not (df["sepal_area"].sub(expected_sepal).abs() < 1e-10).all():
            raise ValueError(f"{path}: incorrect sepal_area values")

        if not (df["petal_area"].sub(expected_petal).abs() < 1e-10).all():
            raise ValueError(f"{path}: incorrect petal_area values")

        comparison = df[original.columns]

    else:
        if list(df.columns) != list(original.columns):
            raise ValueError(
                f"{path}: unexpected columns or column order\n"
                f"Expected: {list(original.columns)}\n"
                f"Actual:   {list(df.columns)}"
            )

        comparison = df

    if not comparison.equals(original):
        raise ValueError(
            f"{path}: values or row order differ from the source dataset"
        )

    print(f"\nFile: {path}")
    print(f"SHA-256: {sha256_file(path)}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column names: {list(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print("Result: OK")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "base":
        check_dataset("iris")

    elif mode == "wine":
        check_dataset("iris")
        check_dataset("wine")

    elif mode == "iris_features":
        check_dataset("iris", expected_features=True)

    elif mode == "all":
        check_dataset("iris", expected_features=True)
        check_dataset("wine")

    else:
        raise ValueError(
            "Usage: python check_data.py "
            "[base|wine|iris_features|all]"
        )


if __name__ == "__main__":
    main()
