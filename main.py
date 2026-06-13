from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from model_reporting import (
    evaluate_and_save_reports,
    prepare_output_dirs,
    save_dataset_preview,
    save_model_with_timestamp,
    save_training_history_plot,
)

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "outputs"

IMAGE_SIZE = (416, 416)
IMAGE_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 100

SHUFFLE_BUFFER_SIZE = 50
PREFETCH_BUFFER_SIZE = 1

RESCALE_FACTOR = 1.0 / 255.0
CONV_FILTERS = (16, 32, 64)
CONV_KERNEL_SIZE = 3
CONV_PADDING = "same"
CONV_ACTIVATION = "relu"
DROPOUT_RATE = 0.2
DENSE_UNITS = 128
DENSE_ACTIVATION = "relu"

METRICS = ["accuracy"]
LOSS_FROM_LOGITS = True
MODEL_PREFIX = "banana_model"

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_file_paths = test_ds.file_paths

class_names = train_ds.class_names
print(class_names)

models_dir, metrics_dir = prepare_output_dirs(OUTPUT_DIR)

save_dataset_preview(train_ds, class_names, OUTPUT_DIR)

train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
val_ds = val_ds.prefetch(PREFETCH_BUFFER_SIZE)
test_ds = test_ds.prefetch(PREFETCH_BUFFER_SIZE)

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(RESCALE_FACTOR, input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)),
    layers.Conv2D(
        CONV_FILTERS[0],
        CONV_KERNEL_SIZE,
        padding=CONV_PADDING,
        activation=CONV_ACTIVATION,
    ),
    layers.MaxPooling2D(),
    layers.Conv2D(
        CONV_FILTERS[1],
        CONV_KERNEL_SIZE,
        padding=CONV_PADDING,
        activation=CONV_ACTIVATION,
    ),
    layers.MaxPooling2D(),
    layers.Conv2D(
        CONV_FILTERS[2],
        CONV_KERNEL_SIZE,
        padding=CONV_PADDING,
        activation=CONV_ACTIVATION,
    ),
    layers.MaxPooling2D(),
    layers.Dropout(DROPOUT_RATE),
    layers.GlobalAveragePooling2D(),
    layers.Dense(DENSE_UNITS, activation=DENSE_ACTIVATION),
    layers.Dense(num_classes),
])

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=SparseCategoricalCrossentropy(from_logits=LOSS_FROM_LOGITS),
    metrics=METRICS,
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr]
)

model_path = save_model_with_timestamp(model, models_dir, prefix=MODEL_PREFIX)
print(f"Model zapisany do: {model_path}")

save_training_history_plot(history, EPOCHS, OUTPUT_DIR)

evaluate_and_save_reports(model, test_ds, test_file_paths, class_names, metrics_dir)
print(f"Metryki zapisane do: {metrics_dir}")
