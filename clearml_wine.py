
from clearml import Dataset

BASE_ID = "2c15801472e0431a901bfe6c4fe0c581"

dataset = Dataset.create(
    dataset_name="sklearn-tabular-bychkova",
    dataset_project="MIIT/MLOps/practice_3/bychkova",
    dataset_version="1.1.0",
    description="Wine dataset added to the base Iris dataset",
    parent_datasets=[BASE_ID],
)

# Добавляем только новый файл Wine
dataset.add_files(path="data/wine.csv")

dataset.upload()
WINE_ID = dataset.id
dataset.finalize()

print("\nWine dataset created successfully")
print("Version: 1.1.0")
print(f"BASE_ID: {BASE_ID}")
print(f"WINE_ID: {WINE_ID}")


