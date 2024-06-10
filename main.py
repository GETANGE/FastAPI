from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
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

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: rate_limit_exceeded_handler(request, exc))
app.add_middleware(SlowAPIMiddleware)

# Path to the saved model file
saved_model_path = 'Models/AppleDiseaseModel1.h5'

# Load the model
model = tf.keras.models.load_model(saved_model_path)

# Define class names
CLASS_NAMES = ['Apple scab', 'Apple healthy']

@app.get("/ping")
@limiter.limit("5/minute")  # Apply rate limit
async def ping(request: Request):
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
@limiter.limit("3/minute")  # Apply rate limit
async def predict(request: Request, file: UploadFile = File(...)):
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

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
    return response

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
