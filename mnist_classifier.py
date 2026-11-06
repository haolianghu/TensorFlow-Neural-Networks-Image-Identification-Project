"""Train and evaluate a reproducible convolutional MNIST classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def prepare_images(images):
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("MNIST images must have shape (n, 28, 28).")
    if images.dtype.kind not in "ui" or images.min(initial=0) < 0 or images.max(initial=0) > 255:
        raise ValueError("Input pixels must be integers between 0 and 255.")
    return (images.astype(np.float32) / 255.0)[..., np.newaxis]


def classification_summary(actual, predicted, probabilities):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    probabilities = np.asarray(probabilities)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Actual and predicted labels must be matching one-dimensional arrays.")
    if probabilities.ndim != 2 or probabilities.shape != (len(actual), 10):
        raise ValueError("Class probabilities must have shape (n, 10).")
    if not len(actual) or not np.isfinite(probabilities).all():
        raise ValueError("Labels and probabilities must be nonempty and finite.")
    if not np.isin(actual, range(10)).all() or not np.isin(predicted, range(10)).all():
        raise ValueError("MNIST labels must be integers from 0 to 9.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to 1.")
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=range(10), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=range(10)).tolist(),
        "top_2_accuracy": float(np.mean(np.any(np.argsort(probabilities, axis=1)[:, -2:] == actual[:, None], axis=1))),
    }


def train(epochs=5, seed=42, model_path=None):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epochs must be a positive integer.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install project dependencies from requirements.txt.") from exc

    tf.keras.utils.set_random_seed(seed)
    (images, labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    images, test_images = prepare_images(images), prepare_images(test_images)
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25, seed=seed),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    history = model.fit(
        train_images, train_labels, validation_data=(validation_images, validation_labels),
        epochs=epochs, batch_size=128, callbacks=[callback], verbose=2,
    )
    probabilities = model.predict(test_images, verbose=0)
    summary = classification_summary(test_labels, probabilities.argmax(axis=1), probabilities)
    if model_path is not None:
        model.save(model_path)
    return {
        "seed": seed,
        "train_rows": len(train_labels),
        "validation_rows": len(validation_labels),
        "test_rows": len(test_labels),
        "epochs_run": len(history.history["loss"]),
        "test_metrics": summary,
        "saved_model": str(model_path) if model_path is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, help="Optional path for the trained Keras model.")
    args = parser.parse_args()
    try:
        result = train(args.epochs, args.seed, args.model)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
