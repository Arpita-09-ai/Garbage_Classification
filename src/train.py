import os
import tensorflow as tf
import matplotlib.pyplot as plt
from src.config import (
    SPLIT_DATASET_PATH,
    SAVED_MODELS_PATH,
    MODEL_NAME,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES
)
from src.model import build_model

# --- Load Data ---
train_path = os.path.join(SPLIT_DATASET_PATH, "train")
val_path = os.path.join(SPLIT_DATASET_PATH, "val")

train_ds = tf.keras.utils.image_dataset_from_directory( # pyright: ignore[reportAttributeAccessIssue]
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory( # pyright: ignore[reportAttributeAccessIssue]
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) # pyright: ignore[reportAttributeAccessIssue]
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) # pyright: ignore[reportAttributeAccessIssue]

# --- Build & Train Model ---
model = build_model(
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    learning_rate=LEARNING_RATE
)
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save Model
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
model_save_path = os.path.join(SAVED_MODELS_PATH, MODEL_NAME)
model.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")

# --- Plot Accuracy & Loss ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
