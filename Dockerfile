# Use the existing TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest

WORKDIR /code

# Copy requirements file
COPY ./requirements.txt /code/requirements.txt

# Install additional dependencies

RUN pip install --upgrade pip  # Ensure pip is up-to-date
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application code
COPY . /code

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
