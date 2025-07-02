import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
import os

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

train_dir = 'dataset/Train'
val_dir = 'dataset/Validation'
test_dir = 'dataset/Test'
model_path = 'model/efficientnet_model.h5'

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='sparse')
val_data = val_gen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='sparse')
test_data = val_gen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', shuffle=False)

base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stop])

if not os.path.exists('model'):
    os.makedirs('model')
model.save(model_path)

loss, acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {acc:.4f}")
