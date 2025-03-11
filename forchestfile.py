import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Step 1: Load image paths and labels
def load_image_paths_and_labels(folder):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                label = os.path.basename(root)
                labels.append(label)
    return np.array(image_paths), np.array(labels)

# Step 2: Create a generator to load images in batches
def image_generator(image_paths, labels, batch_size=32, target_size=(128, 128)):
    num_samples = len(image_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_paths = image_paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            images = []
            for image_path in batch_paths:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, target_size)
                    image = image / 255.0  # Normalize pixel values
                    images.append(image)
            yield np.array(images), np.array(batch_labels)

# Step 3: Load data
folder_path = "C:/Users/sarve/HackAI/AI-1/ds/chest"
image_paths, labels = load_image_paths_and_labels(folder_path)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Split the data
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, y_encoded, test_size=0.2, random_state=42)

# Create generators
train_generator = image_generator(X_train_paths, y_train, batch_size=32)
test_generator = image_generator(X_test_paths, y_test, batch_size=32)

# Step 4: Build the model
def create_transfer_learning_model(input_shape=(128, 128, 3), num_classes=4):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_transfer_learning_model(num_classes=len(label_encoder.classes_))
model.summary()

history = model.fit(train_generator, steps_per_epoch=len(X_train_paths) // 32, epochs=10, validation_data=test_generator, validation_steps=len(X_test_paths) // 32)

loss, accuracy = model.evaluate(test_generator, steps=len(X_test_paths) // 32)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the model
model.save("chest_disease_detection_model.h5")

# Step 8: Predict on a new image
# def predict_new_image(image_path, model, label_encoder, target_size=(128, 128)):
#     image = cv2.imread(image_path)
#     if image is not None:
#         image = cv2.resize(image, target_size)
#         image = image / 255.0  # Normalize pixel values
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
#         prediction = model.predict(image)
#         predicted_class_index = np.argmax(prediction, axis=1)
#         predicted_class = label_encoder.inverse_transform(predicted_class_index)
#         return predicted_class[0]
#     else:
#         return "Error: Image not found or could not be loaded."
#
# # Example usage for prediction
# new_image_path = "path_to_new_pneumonia_image.jpg"
# predicted_class = predict_new_image(new_image_path, model, label_encoder)
# print(f"Predicted Class: {predicted_class}")