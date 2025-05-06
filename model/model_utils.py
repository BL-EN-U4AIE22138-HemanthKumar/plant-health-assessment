import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input # Or your specific model's preprocess
import cv2 # Needed if input is numpy array

# --- Global Model Variable ---
# Load the model once when the application starts
# Adjust the path and loading method based on your model framework (Keras, PyTorch, etc.)
try:
    # Assuming a Keras HDF5 model file
    model = load_model('model/plant_health_assesment_RESNET50.h5')
    print("ResNet50 model loaded successfully.")
    # Define expected input shape (adjust based on your trained model)
    # For ResNet50, common input sizes are (224, 224, 3)
    INPUT_SHAPE = (224, 224)
    # Define class labels in the order the model outputs them
    CLASS_LABELS = ['healthy', 'not_healthy'] # IMPORTANT: Match training order
except Exception as e:
    print(f"Error loading model: {e}")
    print("Prediction functionality will be disabled.")
    model = None
    INPUT_SHAPE = None
    CLASS_LABELS = []


def predict_patch_health(img_array):
    """
    Predicts the health label for a single cropped image patch (numpy array).

    Args:
        img_array (np.ndarray): The cropped image patch in BGR format (from OpenCV).

    Returns:
        str: The predicted label ('healthy', 'not_healthy', or 'error').
    """
    if model is None or INPUT_SHAPE is None or not CLASS_LABELS:
        print("Model not loaded, cannot predict.")
        return 'error - model not loaded'

    try:
        # 1. Preprocess the image array
        # Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Resize to the target input shape expected by the model
        img_resized = cv2.resize(img_rgb, INPUT_SHAPE, interpolation=cv2.INTER_AREA) # Use INTER_AREA for downscaling

        # Convert to Keras image format and add batch dimension
        img_keras_array = image.img_to_array(img_resized)
        img_batch = np.expand_dims(img_keras_array, axis=0)

        # Apply model-specific preprocessing (e.g., normalization)
        img_preprocessed = preprocess_input(img_batch) # Use the correct preprocess_input for your model

        # 2. Predict
        prediction = model.predict(img_preprocessed)

        # 3. Decode prediction
        # Assuming model outputs probabilities for each class (like with softmax)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        if predicted_class_index < len(CLASS_LABELS):
            return CLASS_LABELS[predicted_class_index]
        else:
            print(f"Error: Predicted index {predicted_class_index} out of bounds for labels.")
            return 'error - prediction index error'

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 'error - prediction failed'

# Example of how you might call this from app.py:
# health_results = []
# for i, patch_img_array in enumerate(cropped_patches_data):
#     label = predict_patch_health(patch_img_array)
#     health_results.append((i + 1, label))
