from pathlib import Path
import tempfile

from clearml import Dataset

from prepare_data import load_dataset, save_csv


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        iris_path = Path(temp_dir) / "iris.csv"

        iris = load_dataset("iris")
        save_csv(iris, iris_path)

        print("Original Iris prepared successfully")
        print(f"Rows: {len(iris)}")
        print(f"Columns: {len(iris.columns)}")
        print(f"CSV path: {iris_path}")

        print(
            "\nDataset version 1.0.1 already exists in ClearML. "
            "Not creating a duplicate."
        )


if __name__ == "__main__":
    main()