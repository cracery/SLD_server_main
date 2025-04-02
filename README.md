# Stress Detection API

Проект FastAPI для визначення рівня стресу на основі аналізу емоцій з зображень обличчя.

## Опис проекту

Цей проект надає API для визначення рівня стресу (Low, Middle, High) на основі аналізу емоцій обличчя на зображенні. Система використовує модель DeepFace для розпізнавання емоцій та власну нейронну мережу для прогнозування рівня стресу на основі емоційного вектора.

## Особливості

- API для завантаження зображень і отримання прогнозування рівня стресу
- Підтримка завантаження зображень у форматі файлу або Base64
- Можливість передачі готового вектора емоцій для прогнозування
- Повністю підготовлено для розгортання на render.com

## Структура проекту

```
stress-detection-api/
├── main.py              # Основний файл FastAPI додатку
├── model_loader.py      # Модуль для завантаження моделі нейронної мережі
├── requirements.txt     # Залежності Python
├── Dockerfile           # Докер-файл для створення контейнера
├── render.yaml          # Конфігурація для розгортання на render.com
└── models/              # Папка, куди завантажуються моделі
```

## Вимоги

- Python 3.9+
- FastAPI
- DeepFace
- PyTorch
- OpenCV
- Інші залежності вказані в requirements.txt

## Налаштування та запуск

### Локальний запуск

1. Клонуйте репозиторій:
   ```
   git clone https://github.com/yourusername/stress-detection-api.git
   cd stress-detection-api
   ```

2. Створіть віртуальне середовище та встановіть залежності:
   ```
   python -m venv venv
   source venv/bin/activate  # На Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Запустіть сервер:
   ```
   uvicorn main:app --reload
   ```

### Запуск з Docker

1. Зберіть Docker образ:
   ```
   docker build -t stress-detection-api .
   ```

2. Запустіть контейнер:
   ```
   docker run -p 8000:8000 stress-detection-api
   ```

## Розгортання на Render.com

1. Створіть новий веб-сервіс та вкажіть URL репозиторію.
2. Виберіть "Docker" як середовище.
3. Додайте необхідні змінні середовища:
   - `MODEL_BASE_URL`: URL для завантаження моделей.
4. Натисніть "Create Web Service".

Альтернативно, ви можете використовувати render.yaml для автоматичного розгортання.

## API Endpoints

### `POST /predict/image`

Прогнозує рівень стресу на основі завантаженого зображення.

**Параметри**:
- `file`: Файл зображення

**Відповідь**:
```json
{
  "stress_level": "Low",
  "probabilities": {
    "Low": 0.7,
    "Middle": 0.2,
    "High": 0.1
  },
  "confidence": 0.7
}
```

### `POST /predict/base64`

Прогнозує рівень стресу на основі зображення у форматі base64.

**Параметри**:
- `base64_image`: Base64-кодоване зображення

**Відповідь**: Така ж, як і для `/predict/image`

### `POST /predict/emotions`

Прогнозує рівень стресу на основі вектора емоцій.

**Параметри**:
```json
{
  "emotions": [0.1, 0.0, 0.2, 0.3, 0.1, 0.0, 0.3, 0.0]
}
```
де значення відповідають емоціям: angry, disgust, fear, happy, sad, surprise, neutral, contempt.

**Відповідь**: Така ж, як і для `/predict/image`

### `GET /healthcheck`

Перевіряє стан API.

**Відповідь**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /model_info`

Повертає інформацію про завантажену модель.

**Відповідь**:
```json
{
  "model_name": "stress_detector_weighted_nn",
  "metadata": {
    "date_saved": "2023-01-01",
    "model_type": "StressDetectionNN",
    "metrics": {
      "accuracy": 0.75
    }
  }
}
```
# Інструкція для локального тестування проекту

## Крок 1: Підготовка середовища

1. **Клонування репозиторію або створення нової папки проекту**
   ```bash
   mkdir stress-detection-api
   cd stress-detection-api
   ```

2. **Створення структури файлів**
   Створіть наступні файли в папці проекту:
   - `main.py` - основний файл FastAPI
   - `model_loader.py` - оновлена версія з необхідними змінами
   - `utils.py` - допоміжні функції
   - `requirements.txt` - з додатковим пакетом gdown
   - `Dockerfile` - для Docker-контейнера
   - `client_example.py` - для тестування API

3. **Створення віртуального середовища**
   ```bash
   python -m venv venv
   ```

4. **Активація віртуального середовища**
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source venv/bin/activate
     ```

## Крок 2: Встановлення залежностей

```bash
pip install -r requirements.txt
```

## Крок 3: Налаштування доступу до моделей

Існує два способи завантаження моделей для локального тестування:

### Варіант 1: Завантаження моделей вручну

1. Створіть папку `models` в кореневій директорії проекту:
   ```bash
   mkdir -p models
   ```

2. Завантажте файли з Google Drive вручну і збережіть їх у папці `models` з наступними іменами:
   - `stress_detector_weighted_nn.pth`
   - `stress_detector_weighted_nn_metadata.json`
   - `stress_detector_weighted_nn_feature_importance.npy`
   - `stress_detector_weighted_nn_learned_weights.npy`

### Варіант 2: Автоматичне завантаження при першому запуску

Встановіть змінні середовища з ID папки або окремих файлів:

- Windows:
  ```
  set MODEL_FOLDER_ID=1-ZfjNWADgAOxA2up5ddiiihmt4keB6Kv
  ```

- Linux/Mac:
  ```
  export MODEL_FOLDER_ID=1-ZfjNWADgAOxA2up5ddiiihmt4keB6Kv
  ```

Або встановіть окремі ID файлів, якщо ви маєте прямі посилання:
```
export MODEL_PTH_ID=YOUR_PTH_FILE_ID
export MODEL_METADATA_ID=YOUR_METADATA_FILE_ID
export MODEL_FEATURE_IMPORTANCE_ID=YOUR_FEATURE_IMPORTANCE_FILE_ID
export MODEL_LEARNED_WEIGHTS_ID=YOUR_LEARNED_WEIGHTS_FILE_ID
```

## Крок 4: Запуск сервера

```bash
uvicorn main:app --reload
```

Сервер запуститься за адресою http://localhost:8000

## Крок 5: Тестування API

### Через веб-інтерфейс Swagger UI

Відкрийте браузер і перейдіть за посиланням: http://localhost:8000/docs

### За допомогою client_example.py

```bash
python client_example.py --api_url http://localhost:8000 info
python client_example.py --api_url http://localhost:8000 image шлях/до/зображення.jpg
```

### Перевірка статусу системи

```bash
curl http://localhost:8000/healthcheck
```

## Крок 6: Тестування з Docker (опціонально)

Якщо ви хочете протестувати Docker-контейнер локально:

1. **Збірка Docker-образу**
   ```bash
   docker build -t stress-detection-api .
   ```

2. **Запуск контейнера**
   ```bash
   docker run -p 8000:8000 -e MODEL_FOLDER_ID=1-ZfjNWADgAOxA2up5ddiiihmt4keB6Kv stress-detection-api
   ```

## Вирішення можливих проблем

### Проблеми з gdown

Іноді gdown може мати обмеження при завантаженні файлів з Google Drive. Якщо ви зіткнулися з такою проблемою, спробуйте завантажити файли вручну і помістити їх у папку `models`.

### Проблеми з PyTorch

Якщо виникають помилки з torch, переконайтеся, що ви встановили версію, сумісну з вашою системою:

```bash
pip uninstall torch
pip install torch==2.0.1
```

### Проблеми з доступом до папки Google Drive

Переконайтеся, що папка з моделями налаштована на доступ "Будь-хто з посиланням" (Anyone with the link).

## Ліцензія

MIT