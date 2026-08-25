import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    IMG_SIZE,
    GRADCAM_DIR,
    UPLOAD_DIR
)


def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            return layer.name

        try:

            if len(layer.output.shape) == 4:
                return layer.name

        except Exception:
            continue

    raise ValueError(
        "No suitable convolutional layer found."
    )


def make_gradcam_heatmap(
    image,
    model,
    last_conv_layer_name,
    pred_index=None
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image
        )

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


def generate_gradcam(
    image_path,
    model,
    output_filename
):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    input_image = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        input_image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    last_conv_layer = find_last_conv_layer(
        model
    )

    heatmap = make_gradcam_heatmap(
        input_image,
        model,
        last_conv_layer,
        predicted_index
    )

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy()

    heatmap = np.squeeze(
        heatmap
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image
    )

    plt.imshow(
        heatmap,
        alpha=0.45
    )

    plt.axis("off")

    output_path = os.path.join(
        UPLOAD_DIR,
        output_filename
    )
    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return output_path