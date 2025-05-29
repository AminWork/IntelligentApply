# Use an official Python runtime as a parent image
FROM docker.arvancloud.ir/python:3.12-slim

WORKDIR /app

# Copy & install dependencies (including chainlit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make sure container listens on 8000
EXPOSE 8000

# Start the Chainlit app
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
