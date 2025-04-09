FROM python:3.9-slim

WORKDIR /app

# Встановлення необхідних системних залежностей
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

# Оновлення pip
RUN pip install --upgrade pip

# Копіювання requirements.txt 
COPY requirements.txt .

# Встановлення залежностей з requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Встановлення PyTorch з індексу CPU (це не впливає на версію, тільки на джерело)
RUN pip install --no-cache-dir --force-reinstall torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Створення директорій
RUN mkdir -p /root/.deepface/weights
RUN mkdir -p models

# Завантаження моделі DeepFace
#RUN wget -O /root/.deepface/weights/emotion_model.hdf5 https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5
COPY ./models/facial_expression_model_weights.h5 /root/.deepface/weights/
COPY ./models/* /app/models/

# Копіювання файлів проекту
COPY main.py .
COPY model_loader.py .
COPY utils.py .
COPY *.html .
COPY QR_SLD.png .

# Встановлення змінних середовища
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PORT=8000
EXPOSE 8000

# Запуск FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
