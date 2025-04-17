FROM python:3.9-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# update pip
RUN pip install --upgrade pip

# copy requirements.txt 
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# install PyTorch with CPU index
RUN pip install --no-cache-dir --force-reinstall torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Create directories
RUN mkdir -p /root/.deepface/weights
RUN mkdir -p models

# Download DeepFace model
COPY ./models/facial_expression_model_weights.h5 /root/.deepface/weights/
COPY ./models/* /app/models/

# Copy project files
COPY main.py .
COPY model_loader.py .
COPY utils.py .
COPY static /app/static

# install environment values 
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PORT=8000
EXPOSE 8000

# Run FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
