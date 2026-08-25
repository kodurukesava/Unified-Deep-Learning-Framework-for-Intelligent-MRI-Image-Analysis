import os

from src.predict import MRIPredictor


# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

model = MRIPredictor()


# ==========================================
# 2. FIND YOUR TESTING FOLDER
# ==========================================

project_dir = os.path.dirname(
    os.path.abspath(__file__)
)

testing_folder = os.path.join(
    project_dir,
    "dataset",
    "raw",
    "Testing"
)

print("\nTesting folder:")
print(testing_folder)


# ==========================================
# 3. CHECK THAT TESTING FOLDER EXISTS
# ==========================================

if not os.path.exists(testing_folder):

    raise FileNotFoundError(
        "\nERROR: Testing folder was not found.\n"
        f"Expected location:\n{testing_folder}\n\n"
        "Check that your dataset is inside:\n"
        "dataset/raw/Testing/"
    )


# ==========================================
# 4. FIND AN ACTUAL IMAGE AUTOMATICALLY
# ==========================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_path = None

for root, folders, files in os.walk(
    testing_folder
):

    folders.sort()
    files.sort()

    for filename in sorted(files):

        if filename.lower().endswith(
            valid_extensions
        ):

            image_path = os.path.join(
                root,
                filename
            )

            break

    if image_path is not None:
        break


# ==========================================
# 5. MAKE SURE AN IMAGE WAS FOUND
# ==========================================

if image_path is None:

    raise FileNotFoundError(
        "\nERROR: No image was found inside:\n"
        f"{testing_folder}\n\n"
        "Make sure Testing contains "
        "your MRI image folders and images."
    )


# ==========================================
# 6. PRINT THE REAL IMAGE PATH
# ==========================================

print("\nActual MRI image found:")
print(image_path)


# ==========================================
# 7. RUN PREDICTION
# ==========================================

result = model.predict(
    image_path
)


# ==========================================
# 8. DISPLAY RESULT
# ==========================================

print("\n")
print("=" * 60)
print("             MRI PREDICTION RESULT")
print("=" * 60)

print(
    "\nImage:",
    image_path
)

print(
    "\nPrediction:",
    result["prediction"]
)

print(
    "Confidence:",
    result["confidence"],
    "%"
)

print("\nClass Probabilities:")

for class_name, probability in result[
    "probabilities"
].items():

    print(
        f"  {class_name}: "
        f"{probability}%"
    )

print("=" * 60)