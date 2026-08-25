import os
import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    MODEL_PATH,
    METRICS_DIR,
    IMG_SIZE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS
)

from preprocessing import (
    load_datasets,
    get_augmentation
)

from models.model_architecture import (
    build_model,
    compile_model
)


def train():

    print("=" * 60)
    print("MRI DEEP LEARNING MODEL TRAINING")
    print("=" * 60)

    (
        train_ds,
        validation_ds,
        test_ds,
        class_names
    ) = load_datasets()

    num_classes = len(class_names)

    print("\nDetected Classes:")

    for index, name in enumerate(class_names):
        print(index, "->", name)

    print("\nNumber of classes:", num_classes)

    model, base_model = build_model(
        num_classes=num_classes,
        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        )
    )

    augmentation = get_augmentation()

    train_ds = train_ds.map(
        lambda x, y: (
            augmentation(x, training=True),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    model = compile_model(
        model,
        learning_rate=0.0001
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\nPhase 1: Transfer Learning")

    history1 = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    print("\nPhase 2: Fine Tuning")

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = compile_model(
        model,
        learning_rate=1e-5
    )

    history2 = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    evaluate_model(
        model,
        test_ds,
        class_names
    )

    save_training_graphs(
        history1,
        history2
    )

    print("\nTraining completed successfully.")


def evaluate_model(
    model,
    test_ds,
    class_names
):

    print("\nEvaluating model on TEST dataset...")

    loss, accuracy, precision, recall = model.evaluate(
        test_ds,
        verbose=1
    )

    print("\nTest Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            np.argmax(
                labels.numpy(),
                axis=1
            )
        )

        y_pred.extend(
            np.argmax(
                predictions,
                axis=1
            )
        )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_path = os.path.join(
        METRICS_DIR,
        "classification_report.json"
    )

    with open(
        report_path,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "MRI Classification Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            METRICS_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()


def save_training_graphs(
    history1,
    history2
):

    accuracy = (
        history1.history["accuracy"]
        +
        history2.history["accuracy"]
    )

    val_accuracy = (
        history1.history["val_accuracy"]
        +
        history2.history["val_accuracy"]
    )

    loss = (
        history1.history["loss"]
        +
        history2.history["loss"]
    )

    val_loss = (
        history1.history["val_loss"]
        +
        history2.history["val_loss"]
    )

    epochs = range(
        1,
        len(accuracy) + 1
    )

    # Accuracy Graph

    plt.figure()

    plt.plot(
        epochs,
        accuracy,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "MRI Model Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            METRICS_DIR,
            "accuracy.png"
        )
    )

    plt.close()

    # Loss Graph

    plt.figure()

    plt.plot(
        epochs,
        loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "MRI Model Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            METRICS_DIR,
            "loss.png"
        )
    )

    plt.close()


if __name__ == "__main__":

    train()