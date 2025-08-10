import tensorflow as tf
from tensorflow.keras import layers, models, Input # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import EfficientNetV2B2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]

def build_model(input_shape, num_classes, learning_rate):
    # Data augmentation layer
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # Base model: EfficientNetV2B2 pre-trained on ImageNet
    base_model = EfficientNetV2B2(include_top=False, input_shape=input_shape, weights="imagenet")
    
    # Fine-tune: freeze the first 200 layers
    base_model.trainable = True
    for layer in base_model.layers[:200]:
        layer.trainable = False

    # Full model with explicit Input layer
    model = models.Sequential([
        Input(shape=input_shape),   # ✅ Fix: ensures model is "built"
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model built and compiled successfully.")
    return model
