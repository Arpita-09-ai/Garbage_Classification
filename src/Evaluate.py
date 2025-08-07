import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import SPLIT_DATASET_PATH, IMG_SIZE, BATCH_SIZE,SAVED_MODELS_PATH, MODEL_NAME

# --- 1. Load the saved model and the test data ---

model_load_path = os.path.join(SAVED_MODELS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_load_path)
test_path = os.path.join(SPLIT_DATASET_PATH, "test")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False  # Important: Do not shuffle for evaluation
)
class_names = test_ds.class_names
print(f"Class names: {class_names}")


# --- 2. Get predictions and true labels ---

y_true = []
y_pred_probs = []

# Iterate over the test dataset
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(preds)

# Convert probabilities to class indices
y_pred = np.argmax(y_pred_probs, axis=1)


# --- 3. Generate and display the confusion matrix ---
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# 


# --- 4. Generate and print the classification report ---
print("\nGenerating classification report...")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)