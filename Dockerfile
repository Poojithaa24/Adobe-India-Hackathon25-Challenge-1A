# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files
COPY . .

# Install system-level dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1-mesa-glx libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Create input and output directories (in case not present in image)
RUN mkdir -p input output

# Run the main script
CMD ["python", "main.py"]
