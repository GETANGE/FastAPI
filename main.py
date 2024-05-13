from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Keeping CORS middleware for potential cross-origin requests
app = FastAPI()

origins = ["*"]  # Allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the saved model directory
saved_model_dir = 'Models/1'

# Load the model using TFSMLayer
model_layer = tf.keras.layers.TFSMLayer(saved_model_dir, call_endpoint='serving_default')

CLASS_NAMES = ['Apple Apple scab', 'Apple healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    try:
        # Read the image file
        image = read_file_as_image(await file.read())

        # Prepare the image for prediction (might differ based on SavedModel input)
        img_batch = np.expand_dims(image, 0)  # Assuming batch size of 1

        # Make prediction using the model layer
        predictions = model_layer(img_batch)

        # Convert predictions tensor to NumPy array
        predictions_array = predictions['dense_1'].numpy()

        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(predictions_array)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions_array[0][predicted_class_index])  # Access confidence value

        # Check if the predicted class is an apple leaf
        if "Apple" not in predicted_class:
            predicted_class = "Not an apple leaf"
            confidence = 0.0

        # Now, create the response dictionary
        response = {
            "Predicted": predicted_class,
            "confidence": confidence
        }

        # Return the response
        return response

    except Exception as e:
        # Handle any errors that occur during prediction
        error_message = f"Error during prediction: {str(e)}"
        # Return an error response
        return {"error": error_message}


if __name__ == "__main__":
    # Get the port number from the environment variable or use a default value
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='localhost', port=port)
