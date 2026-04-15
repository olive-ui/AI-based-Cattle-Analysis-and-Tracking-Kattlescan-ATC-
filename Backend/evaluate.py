import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

DATASET_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\AD BACKEND\animal-atc-backend\Indian_bovine_breeds\Indian_bovine_breeds"

# load saved model
model = tf.keras.models.load_model("best_cnn_model.keras")

# load validation data
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# evaluate
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"\nAccuracy: {accuracy * 100:.1f}%")
print(f"Loss: {loss:.4f}")