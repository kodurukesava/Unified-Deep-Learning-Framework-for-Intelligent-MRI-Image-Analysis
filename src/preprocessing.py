import os
import json
import tensorflow as tf

from config import (
    RAW_DATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    MODEL_DIR
)


TRAIN_DIR = os.path.join(
    RAW_DATA_DIR,
    "Training"
)

TEST_DIR = os.path.join(
    RAW_DATA_DIR,
    "Testing"
)


def load_datasets():

    print("Training directory:")
    print(TRAIN_DIR)

    print("\nTesting directory:")
    print(TEST_DIR)

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}"
        )

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(
            f"Testing folder not found: {TEST_DIR}"
        )

    # 80% of Training -> actual training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="training",
        seed=RANDOM_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    # 20% of Training -> validation
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    # Existing Testing folder -> final test
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(i, "->", name)

    # Make sure Testing has the same classes
    if test_ds.class_names != class_names:

        raise ValueError(
            "\nTraining and Testing classes do not match.\n"
            f"Training classes: {class_names}\n"
            f"Testing classes: {test_ds.class_names}"
        )

    class_indices = {
        index: name
        for index, name in enumerate(class_names)
    }

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    with open(
        os.path.join(
            MODEL_DIR,
            "class_names.json"
        ),
        "w"
    ) as file:

        json.dump(
            class_indices,
            file,
            indent=4
        )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    validation_ds = validation_ds.prefetch(
        AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        AUTOTUNE
    )

    return (
        train_ds,
        validation_ds,
        test_ds,
        class_names
    )


def get_augmentation():

    augmentation = tf.keras.Sequential([

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )

    ])

    return augmentation