from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models

def build_densenet121(input_shape=(224, 224, 3), num_classes=1):
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(num_classes, activation="sigmoid")(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    return model
