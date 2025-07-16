import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import numpy as np
import os

# Settings
img_size = (224, 224)
num_classes = 5
samples_per_class = 10
save_path = 'butterflyresnet50.hdf5'

# Dummy dataset creation
X = np.random.rand(samples_per_class * num_classes, img_size[0], img_size[1], 3).astype(np.float32)
y = np.repeat(np.arange(num_classes), samples_per_class)
y = tf.keras.utils.to_categorical(y, num_classes)

# Base Model with ResNet50
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights=None)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=8)

# Save the model
model.save(save_path)
print(f"✅ Complex butterflyresnet50.hdf5 model saved at {save_path}")
