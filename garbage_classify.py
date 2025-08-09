import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Paths & configs
data_dir = "garbage-dataset"
img_size = (224, 224)
batch_size = 32
epochs = 15
model_path = "Waste_classifier_v2.h5"

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load or build model
if os.path.exists(model_path):
    print("\n📦 Loading pre-trained model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("\n🔨 Training new model...")
    # Base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Compile and train
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Save model
    model.save(model_path)
    print(f"\n✅ Model trained and saved to {model_path}")

# Show class indices
print("\n📚 Class Indices:")
for cls, idx in train_generator.class_indices.items():
    print(f"{idx}: {cls}")

# Load and predict an image
img_path = "garbage-dataset/clothes/clothes_10.jpg"  # Change path if needed
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
pred_class_idx = np.argmax(pred)
confidence = np.max(pred)

# Get readable class name
class_labels = list(train_generator.class_indices.keys())
predicted_class = class_labels[pred_class_idx]

# Output prediction
print(f"\n🧠 Predicted class: {predicted_class}")
print(f"✅ Confidence: {confidence * 100:.2f}%")
