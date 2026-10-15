from clearml import Dataset

dataset_ids = {
    "1.0.1 — original Iris": "2c15801472e0431a901bfe6c4fe0c581",
    "1.1.0 — Iris + Wine": "5cdf5d16eb0e4b018a11019bffe5a139",
    "2.0.1 — Iris with features": "1d52905ef88d44bca60b0f7c068ee3b8",
}

for label, dataset_id in dataset_ids.items():
    print(f"\n--- {label} ---")
    dataset = Dataset.get(dataset_id=dataset_id)

    print("ID:", dataset.id)
    print("Name:", dataset.name)
    print("Version:", dataset.version)
    print("Files:", sorted(dataset.list_files()))
    print("Dependency graph:", dataset.get_dependency_graph())
