import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Step 1: Load the saved model
model = load_model("chest_disease_detection_model.h5")

# Step 2: Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, target_size)  # Resize image
        image = image / 255.0  # Normalize pixel values
        return image
    return None

# Step 3: Function to make a prediction
def predict_image(model, image_path, target_size=(128, 128)):
    # Preprocess the image
    image = preprocess_image(image_path, target_size)
    if image is not None:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        # Assuming binary classification: 0 = Healthy, 1 = Diseased
        if prediction[0][0] > 0.5:  # Threshold for binary classification
            return "Diseased"
        else:
            return "Healthy"
    return "Invalid Image"

# Step 4: Test the model on a single image
image_path = "chesttest1.jpeg" # Replace with the path to your test image
prediction = predict_image(model, image_path)

# Print the result
print(f"Prediction: {prediction}")
