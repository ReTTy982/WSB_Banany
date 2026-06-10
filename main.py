import os
import numpy as np
import tensorflow as tf
import matplotlib

from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

image_size = (416, 416)
batch_size = 8

train_ds = keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = keras.utils.image_dataset_from_directory(
    "dataset/valid",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
)

test_ds = keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
)

class_names = train_ds.class_names
print(class_names)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    count = min(12, images.shape[0])

    for i in range(count):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")

plt.savefig(os.path.join(output_dir, "dataset_preview.png"), dpi=200, bbox_inches="tight")
plt.close()

train_ds = train_ds.shuffle(50).prefetch(1)
val_ds = val_ds.prefetch(1)
test_ds = test_ds.prefetch(1)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(416, 416, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.GlobalAveragePooling2D(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=200, bbox_inches="tight")
plt.close()

model.evaluate(test_ds)
