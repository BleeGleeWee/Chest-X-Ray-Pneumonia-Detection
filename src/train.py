import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from densenet_model import build_densenet121

# Paths
data_dir = "../data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(224,224), class_mode='binary')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(224,224), class_mode='binary')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224,224), class_mode='binary', shuffle=False)

# Model
model = build_densenet121()
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
os.makedirs("../results", exist_ok=True)
model.save("../results/densenet_pneumonia.h5")

# Evaluate
y_true = test_data.classes
y_probs = model.predict(test_data).ravel()
y_pred = (y_probs > 0.5).astype("int32")

print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
