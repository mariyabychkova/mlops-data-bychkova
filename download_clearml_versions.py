from pathlib import Path
from clearml import Dataset


# Скачиваем первоначальные версии для проверки ошибок.
# Исправленные версии: 1.0.1, 1.1.0 и 2.0.1.
versions = {
    "1.0.0": "070ba818ca5c4ee89465f0aafdbe066b",
    "1.1.0": "224cde49b2734ca2af6a890d0a44955a",
    "2.0.0": "8cba7a996f7640ffbc401bd36b7a74d3",
}

root = Path("clearml_downloads")
root.mkdir(exist_ok=True)

for version, dataset_id in versions.items():
    target = root / version
    target.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Downloading {version} ---")
    dataset = Dataset.get(dataset_id=dataset_id)
    result = dataset.get_mutable_local_copy(
        target_folder=str(target),
        overwrite=True,
    )

    print("Local path:", result)
    print("Files:", sorted(
        str(path.relative_to(target))
        for path in target.rglob("*")
        if path.is_file()
    ))
