import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from src.config import SPLIT_DATASET_PATH, IMG_SIZE, BATCH_SIZE, SAVED_MODELS_PATH, MODEL_NAME


# --- 1. Load the saved model and the test data ---
model_load_path = os.path.join(SAVED_MODELS_PATH, MODEL_NAME)
print(f"Loading model from: {model_load_path}")
model = tf.keras.models.load_model(model_load_path)

test_path = os.path.join(SPLIT_DATASET_PATH, "test")
print(f"Loading test dataset from: {test_path}")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)
class_names = test_ds.class_names # pyright: ignore[reportAttributeAccessIssue]
print(f"Class names: {class_names}")

# --- 2. Get predictions and true labels ---
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0) # pyright: ignore[reportOptionalMemberAccess]
    y_true.extend(labels.numpy())
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- 3. Confusion Matrix ---
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- 4. Classification Report ---
print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# --- 5. Average Confidence Per Class ---
print("\nAverage confidence per class:")
for i, class_name in enumerate(class_names):
    class_indices = np.where(y_pred == i)[0]
    if len(class_indices) > 0:
        avg_conf = np.mean(y_pred_probs[class_indices, i])
        print(f"{class_name}: {avg_conf:.4f}")
    else:
        print(f"{class_name}: No predictions made for this class.")

# --- 6. Save per-image predictions with confidence ---
file_paths = test_ds.file_paths # pyright: ignore[reportAttributeAccessIssue]
confidences = np.max(y_pred_probs, axis=1) * 100  # in %
pred_labels = [class_names[i] for i in y_pred]
true_labels = [class_names[i] for i in y_true]

results_df = pd.DataFrame({
    "file_path": file_paths,
    "true_label": true_labels,
    "predicted_label": pred_labels,
    "confidence_score(%)": confidences
})

csv_path = "test_predictions_with_confidence.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Saved per-image predictions with confidence to: {csv_path}")

# --- 7. Plot training history if available ---
history_path = os.path.join(SAVED_MODELS_PATH, "history.json")
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
else:
    print("\nNo training history found. Skipping accuracy/loss plot.")
print("\nSample predictions with confidence scores:")
print(results_df.head(10))  # Shows first 10 rows of the predictions CSV in console
