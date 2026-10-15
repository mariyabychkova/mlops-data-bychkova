
from clearml import Dataset

BASE_ID = "2c15801472e0431a901bfe6c4fe0c581"


dataset = Dataset.create(
    dataset_name="sklearn-tabular-bychkova",
    dataset_project="MIIT/MLOps/practice_3/bychkova",
    dataset_version="2.0.1",
    description="Iris dataset with sepal_area and petal_area features",
    parent_datasets=[BASE_ID],
)

# Добавляем только изменённый Iris
dataset.add_files(path="data/iris.csv")

dataset.upload()
IRIS_FEATURES_ID = dataset.id
dataset.finalize()

print("\nIris features dataset created successfully")
print("Version: 2.0.1")
print(f"BASE_ID: {BASE_ID}")
print(f"IRIS_FEATURES_ID: {IRIS_FEATURES_ID}")
