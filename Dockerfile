# Base image with TensorFlow Serving
FROM tensorflow/serving:latest

# Set working directory
WORKDIR /app

# Copy model directory
COPY Models /models

# Define the default command (replace with your actual command)
CMD ["tensorflow_serving", "--port=8840"]

# (Optional) Expose port for model serving
EXPOSE 8840
