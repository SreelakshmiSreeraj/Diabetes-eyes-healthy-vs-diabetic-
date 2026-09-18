from pathlib import Path
import csv

# Dataset location
DATASET_PATH = Path(__file__).parent / "dataset" / "Dataset" / "Subjects"

healthy_path = DATASET_PATH / "Healthy Subjects"
diabetes_path = DATASET_PATH / "Diabetes Subjects"

rows = []

# -------------------------
# HEALTHY
# -------------------------
for subject_folder in healthy_path.iterdir():

    if not subject_folder.is_dir():
        continue

    subject_id = subject_folder.name

    # Exclude problematic Subject 19
    if subject_id == "19":
        continue

    for image in subject_folder.iterdir():

        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # Subject 23 has one extra DSC image
        if subject_id == "23" and image.name == "DSC_5407.JPG":
            continue

        rows.append([
            str(image),
            subject_id,
            "Healthy"
        ])


# -------------------------
# DIABETES
# -------------------------
for subject_folder in diabetes_path.iterdir():

    if not subject_folder.is_dir():
        continue

    subject_id = subject_folder.name

    for image in subject_folder.iterdir():

        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # Ignore retinal fundus images
        if image.stem in {f"{subject_id}_L", f"{subject_id}_R"}:
            continue

        rows.append([
            str(image),
            subject_id,
            "Diabetes"
        ])


# -------------------------
# SAVE CSV
# -------------------------
output_file = Path(__file__).parent / "dataset_manifest.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "image_path",
        "subject_id",
        "label"
    ])

    writer.writerows(rows)


print("=" * 50)
print("DATASET MANIFEST CREATED")
print("=" * 50)

print(f"Total images : {len(rows)}")

healthy_count = sum(1 for row in rows if row[2] == "Healthy")
diabetes_count = sum(1 for row in rows if row[2] == "Diabetes")

print(f"Healthy images  : {healthy_count}")
print(f"Diabetes images : {diabetes_count}")
print("=" * 50)