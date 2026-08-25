import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

RAW_DATA_DIR = os.path.join(
    DATASET_DIR,
    "raw"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_resnet50_mri.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PREDICTION_DIR = os.path.join(
    RESULTS_DIR,
    "predictions"
)

GRADCAM_DIR = os.path.join(
    RESULTS_DIR,
    "gradcam_outputs"
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "performance_metrics"
)

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "mri_analysis.db"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

UPLOAD_DIR = os.path.join(
    STATIC_DIR,
    "uploads"
)

IMG_SIZE = 224

CONFIDENCE_THRESHOLD = 0.50


def create_directories():

    directories = [
        MODEL_DIR,
        RESULTS_DIR,
        PREDICTION_DIR,
        GRADCAM_DIR,
        METRICS_DIR,
        DATABASE_DIR,
        UPLOAD_DIR
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )


create_directories()