"""Train the CNN used by detect_gui.py to classify Angry / Happy / Sad faces.

Reads 48x48 grayscale face crops from dataset/train and dataset/test
(class-per-folder layout), trains a small CNN, and saves the result as a
Keras .h5 model compatible with detect_gui.py.

Usage:
    python train.py
    python train.py --epochs 25 --batch-size 64
    python train.py --output emotion_model.h5   # overwrite the shipped model
"""

import argparse
import csv
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

IMG_SIZE = 48
CLASS_NAMES = ["angry", "happy", "sad"]  # index order must match EMOTIONS in detect_gui.py
SEED = 42
# -----------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                         help="Dataset root containing train/ and test/ subfolders")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "emotion_model_trained.h5"),
                         help="Where to save the trained model "
                              "(defaults to a new file so the shipped emotion_model.h5 "
                              "is never overwritten by accident)")
    return parser.parse_args()


def build_datasets(dataset_dir, batch_size):
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    # Class counts -> class weights, since "angry" is under-represented
    # (roughly 1.1k train images vs. ~7.2k for "happy").
    counts = {
        name: len(os.listdir(os.path.join(train_dir, name))) for name in CLASS_NAMES
    }
    total = sum(counts.values())
    class_weight = {
        i: total / (len(CLASS_NAMES) * counts[name]) for i, name in enumerate(CLASS_NAMES)
    }

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, test_ds, class_weight, counts


def build_model():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Rescaling(1.0 / 255),
        data_augmentation,

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="emotion_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = zip(
        range(1, len(history.history["loss"]) + 1),
        history.history["loss"],
        history.history["accuracy"],
        history.history["val_loss"],
        history.history["val_accuracy"],
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        writer.writerows(rows)


def main():
    args = parse_args()

    train_ds, test_ds, class_weight, counts = build_datasets(args.dataset_dir, args.batch_size)
    print(f"Class counts (train): {counts}")
    print(f"Class weights: {class_weight}")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            args.output, monitor="val_accuracy", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    save_history(history, os.path.join(LOG_DIR, "training_history.csv"))

    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\nFinal test accuracy: {test_accuracy * 100:.2f}%  (loss: {test_loss:.4f})")
    print(f"Best model saved to: {args.output}")
    print("To use it in the app, copy/rename it to 'emotion_model.h5'.")


if __name__ == "__main__":
    main()
