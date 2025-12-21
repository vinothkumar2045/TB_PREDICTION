import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ========================
# Paths
# ========================
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# ========================
# Data Generators (Force RGB)
# ========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# ========================
# Model: EfficientNetB0
# ========================
base_model = EfficientNetB0(
    weights=None,             # ❌ No pretrained weights (set to "imagenet" if you want transfer learning)
    include_top=False,
    input_shape=(224, 224, 3) # ✅ Match RGB generators
)

# Freeze base model initially
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ========================
# Compile
# ========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ========================
# Train (Stage 1 - Frozen Base)
# ========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# ========================
# Fine-Tuning (Stage 2 - Unfreeze last layers)
# ========================
base_model.trainable = True
for layer in base_model.layers[:-50]:   # keep most layers frozen
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # smaller LR
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# ========================
# Evaluate
# ========================
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# ========================
# Save model
# ========================
model.save("efficientnet_tb.h5")
print("Model saved as efficientnet_tb.h5 ✅")
