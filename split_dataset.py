from pathlib import Path
import csv
import random

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_PATH = Path(__file__).parent
MANIFEST_PATH = PROJECT_PATH / "dataset_manifest.csv"
OUTPUT_PATH = PROJECT_PATH / "dataset_split.csv"

# --------------------------------------------------
# LOAD MANIFEST
# --------------------------------------------------

with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("=" * 55)
print("SUBJECT-LEVEL DATASET SPLIT")
print("=" * 55)

print(f"Total images   : {len(rows)}")

# --------------------------------------------------
# CREATE UNIQUE SUBJECT IDs
# --------------------------------------------------
# Healthy subject 1 and Diabetes subject 1
# are DIFFERENT people.
#
# Therefore we use:
# Healthy_1
# Diabetes_1

subjects = {}

for row in rows:

    unique_subject_id = f"{row['label']}_{row['subject_id']}"

    row["unique_subject_id"] = unique_subject_id

    subjects[unique_subject_id] = row["label"]

print(f"Total subjects : {len(subjects)}")

# --------------------------------------------------
# SEPARATE BY CLASS
# --------------------------------------------------

healthy_subjects = [
    subject_id
    for subject_id, label in subjects.items()
    if label == "Healthy"
]

diabetes_subjects = [
    subject_id
    for subject_id, label in subjects.items()
    if label == "Diabetes"
]

print("\nSubjects by class:")
print(f"Healthy  : {len(healthy_subjects)}")
print(f"Diabetes : {len(diabetes_subjects)}")

# --------------------------------------------------
# FIXED RANDOM SEED
# --------------------------------------------------

random.seed(42)

random.shuffle(healthy_subjects)
random.shuffle(diabetes_subjects)

# --------------------------------------------------
# SPLIT EACH CLASS 70 / 15 / 15
# --------------------------------------------------

def split_subjects(subject_list):

    total = len(subject_list)

    train_count = round(total * 0.70)
    val_count = round(total * 0.15)

    train = subject_list[:train_count]

    val = subject_list[
        train_count:train_count + val_count
    ]

    test = subject_list[
        train_count + val_count:
    ]

    return train, val, test


healthy_train, healthy_val, healthy_test = split_subjects(
    healthy_subjects
)

diabetes_train, diabetes_val, diabetes_test = split_subjects(
    diabetes_subjects
)

# --------------------------------------------------
# COMBINE
# --------------------------------------------------

train_ids = set(
    healthy_train + diabetes_train
)

val_ids = set(
    healthy_val + diabetes_val
)

test_ids = set(
    healthy_test + diabetes_test
)

# --------------------------------------------------
# ASSIGN SPLITS
# --------------------------------------------------

for row in rows:

    subject_id = row["unique_subject_id"]

    if subject_id in train_ids:
        row["split"] = "train"

    elif subject_id in val_ids:
        row["split"] = "val"

    elif subject_id in test_ids:
        row["split"] = "test"

# --------------------------------------------------
# SAVE
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image_path",
        "subject_id",
        "unique_subject_id",
        "label",
        "split"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nSUBJECT SPLIT")
print("-" * 40)

print(f"Training subjects   : {len(train_ids)}")
print(f"Validation subjects : {len(val_ids)}")
print(f"Testing subjects    : {len(test_ids)}")

print("\nIMAGE SPLIT")
print("-" * 40)

train_images = sum(
    1 for row in rows
    if row["split"] == "train"
)

val_images = sum(
    1 for row in rows
    if row["split"] == "val"
)

test_images = sum(
    1 for row in rows
    if row["split"] == "test"
)

print(f"Training images   : {train_images}")
print(f"Validation images : {val_images}")
print(f"Testing images    : {test_images}")

print("\nCLASS DISTRIBUTION")
print("-" * 40)

for split in ["train", "val", "test"]:

    healthy = sum(
        1 for row in rows
        if row["split"] == split
        and row["label"] == "Healthy"
    )

    diabetes = sum(
        1 for row in rows
        if row["split"] == split
        and row["label"] == "Diabetes"
    )

    print(
        f"{split:5} → "
        f"Healthy: {healthy:3} | "
        f"Diabetes: {diabetes:3}"
    )

print("\n" + "=" * 55)
print("SPLIT COMPLETED SUCCESSFULLY")
print("=" * 55)