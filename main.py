from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Path to the saved model file
saved_model_path = '/home/emmanuel/Desktop/Projects/FastAPI/Models/AppleDiseaseModel1.h5'

# Load the model
model = tf.keras.models.load_model(saved_model_path)

# Define class names
CLASS_NAMES = ['Apple scab', 'Apple healthy']

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

        # Prepare the image for prediction
        img_batch = np.expand_dims(image, 0)  # Assuming batch size of 1

        # Make prediction
        predictions = model.predict(img_batch)

        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = int(predictions[0][predicted_class_index] * 100)  # Convert to whole percentage

        # Create the response dictionary
        response = {
            "Predicted": predicted_class,
            "confidence": f"{confidence}%"  # Format as whole percentage
        }

        # Return the response
        return response

    except Exception as e:
        # Handle any errors that occur during prediction
        error_message = f"An error occurred: {str(e)}"
        # Return an error response
        return {"error": error_message}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
