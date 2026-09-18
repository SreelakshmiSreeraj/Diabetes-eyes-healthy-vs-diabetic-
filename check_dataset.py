from pathlib import Path

# Dataset location
DATASET_PATH = Path(__file__).parent / "dataset" / "Dataset" / "Subjects"

healthy_path = DATASET_PATH / "Healthy Subjects"
diabetes_path = DATASET_PATH / "Diabetes Subjects"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_conjunctival_images(subject_folder, is_diabetes=False):
    """Get only conjunctival images, excluding fundus images."""

    images = [
        file for file in subject_folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if is_diabetes:
        # Fundus images are named like 9_L.jpg and 9_R.jpg
        images = [
            img for img in images
            if img.stem not in {
                f"{subject_folder.name}_L",
                f"{subject_folder.name}_R"
            }
        ]

    return images


def check_group(group_path, label, is_diabetes=False):
    subjects = [
        folder for folder in group_path.iterdir()
        if folder.is_dir()
    ]

    total_images = 0
    problems = []

    print(f"\n{label}")
    print("-" * 40)
    print(f"Number of subjects: {len(subjects)}")

    for subject in subjects:
        images = get_conjunctival_images(subject, is_diabetes)
        total_images += len(images)

        if len(images) != 8:
            problems.append(
                f"Subject {subject.name}: "
                f"{len(images)} conjunctival images"
            )

    print(f"Conjunctival images: {total_images}")

    if problems:
        print("\n⚠️ Problems found:")
        for problem in problems:
            print(problem)
    else:
        print("✅ Every subject has exactly 8 conjunctival images.")

    return len(subjects), total_images


print("=" * 55)
print("       BioLens AI - Dataset Verification")
print("=" * 55)

healthy_subjects, healthy_images = check_group(
    healthy_path,
    "HEALTHY",
    is_diabetes=False
)

diabetes_subjects, diabetes_images = check_group(
    diabetes_path,
    "DIABETES",
    is_diabetes=True
)

print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)

print(f"Healthy subjects       : {healthy_subjects}")
print(f"Diabetes subjects      : {diabetes_subjects}")
print(f"Total subjects         : {healthy_subjects + diabetes_subjects}")

print(f"Healthy images         : {healthy_images}")
print(f"Diabetes images        : {diabetes_images}")
print(f"Total conjunctival     : {healthy_images + diabetes_images}")

print("=" * 55)