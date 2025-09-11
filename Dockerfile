FROM python:3.9-slim-bookworm

WORKDIR /app

# Оновлюємо apt та встановлюємо залежності
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Оновлюємо pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Копіюємо requirements.txt
COPY requirements.txt .

# Встановлюємо залежності з таймаутом та повторними спробами
RUN pip install --no-cache-dir --timeout=100 --retries=5 -r requirements.txt \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Встановлюємо PyTorch (CPU)
RUN pip install --no-cache-dir --force-reinstall torch==2.0.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Створюємо директорії
RUN mkdir -p /root/.deepface/weights \
 && mkdir -p /app/models

# Копіюємо моделі
COPY ./models/facial_expression_model_weights.h5 /root/.deepface/weights/
COPY ./models/* /app/models/

# Копіюємо код проєкту
COPY main.py .
COPY model_loader.py .
COPY utils.py .
COPY static /app/static

# Змінні середовища
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=pythonENV PORT=8000
ENV PORT=8000
EXPOSE 8000

# Run FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

