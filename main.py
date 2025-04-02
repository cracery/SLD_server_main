import os
import base64
import torch
import numpy as np
import io
from PIL import Image
from io import BytesIO
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_307_TEMPORARY_REDIRECT
from pydantic import BaseModel
import cv2
from deepface import DeepFace


# Імпорт власних модулів
from model_loader import load_weighted_model

# Глобальні змінні для моделі
stress_model = None
model_metadata = None
feature_importance = None
learned_weights = None

app = FastAPI(title="Stress Detection API", description="API для визначення рівня стресу на основі аналізу емоцій обличчя.")

# Налаштування CORS для доступу з веб-клієнтів
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Завантаження моделі при запуску сервера
model_name = "stress_detector_weighted_nn"
#model = None
#metadata = None

# Спробуємо підключити статичні файли, якщо вони є
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    # Якщо директорії немає, просто продовжуємо
    pass

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Головна сторінка сервісу. За замовчуванням повертає analyse.html
    """
    return default_html_response()

@app.get("/analyse", response_class=HTMLResponse)
async def get_analyse_html():
    """
    Відкриває файл analyse.html з аналізом стресу
    """
    try:
        with open("analyse.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return default_html_response()

@app.get("/speedometer", response_class=HTMLResponse)
async def get_speedometer_html():
    """
    Відкриває файл speedometer.html зі спідометром рівня стресу
    """
    try:
        with open("speedometer.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return RedirectResponse(url="/", status_code=HTTP_307_TEMPORARY_REDIRECT)

@app.get("/docs-redirect")
async def redirect_to_docs():
    """
    Перенаправляє користувача на сторінку документації API
    """
    return RedirectResponse(url="https://stress-detection-api-ym1t.onrender.com/docs#", 
                           status_code=HTTP_307_TEMPORARY_REDIRECT)

def default_html_response():
    """
    Повертає стандартний HTML у випадку, якщо файл не знайдено
    """
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Stress Detection API</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                .nav-links {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .nav-links a {
                    display: inline-block;
                    margin-right: 15px;
                    text-decoration: none;
                    color: #3498db;
                    font-weight: bold;
                }
                .nav-links a:hover {
                    color: #2980b9;
                    text-decoration: underline;
                }
                ul {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                }
                li {
                    margin-bottom: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Stress Detection API</h1>
            
            <div class="nav-links">
                <a href="/">Головна</a>
                <a href="/analyse">Аналіз стресу</a>
                <a href="/speedometer">Спідометр</a>
                <a href="/docs-redirect">API Документація</a>
            </div>
            
            <p>API для визначення рівня стресу на основі аналізу емоцій обличчя.</p>
            <p>Доступні ендпоінти:</p>
            <ul>
                <li><strong>/predict/image</strong> - POST-запит для аналізу зображення</li>
                <li><strong>/predict/base64</strong> - POST-запит для аналізу зображення в base64</li>
                <li><strong>/predict/emotions</strong> - POST-запит для аналізу вектора емоцій</li>
                <li><strong>/healthcheck</strong> - GET-запит для перевірки стану сервісу</li>
                <li><strong>/model_info</strong> - GET-запит для отримання інформації про модель</li>
            </ul>
            
            <p>Для перегляду детальної документації API відвідайте <a href="https://stress-detection-api-ym1t.onrender.com/docs#">документацію</a>.</p>
        </body>
    </html>
    """)

# Перевірка, чи DeepFace працює правильно
try:
    # Створюємо фіктивне зображення для тесту
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_result = DeepFace.analyze(img_path=dummy_img, actions=['emotion'], enforce_detection=False)
    print("DeepFace успішно ініціалізовано")
except Exception as e:
    print(f"Помилка при ініціалізації DeepFace: {str(e)}")
    import traceback
    print(traceback.format_exc())

@app.on_event("startup")
async def startup_event():
    #global model, metadata
    global model_name, stress_model, model_metadata, feature_importance, learned_weights
    try:
        print("Завантаження моделі при запуску...")
        stress_model, feature_importance, learned_weights, model_metadata = load_weighted_model(model_name)
        print("Модель успішно завантажена")
    except Exception as e:
        print(f"Помилка при завантаженні моделі: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("Перевірка DeepFace...")
    try:
        import os
        # Перевіряємо шлях до моделі
        model_path = os.path.join(os.path.expanduser('~'), '.deepface', 'weights', 'facial_expression_model_weights.h5')
        if os.path.isfile(model_path):
            print(f"Модель DeepFace знайдена за шляхом {model_path}")
        else:
            print(f"Модель DeepFace не знайдена за шляхом {model_path}")
            # Альтернативний шлях
            if os.path.isfile("/root/.deepface/weights/efacial_expression_model_weights.h5"):
                print("Модель знайдена за альтернативним шляхом")
            else:
                print("Модель не знайдена за альтернативним шляхом")
    except Exception as e:
        print(f"Помилка при перевірці DeepFace: {e}")

class EmotionVector(BaseModel):
    emotions: List[float]

class StressResponse(BaseModel):
    stress_level: str
    probabilities: dict
    confidence: float

def generate_emotion_vector(image_data):
    """
    Генерує вектор емоцій за допомогою моделі DeepFace.
    """
    try:
        # Конвертація масиву байтів у зображення
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

            # Якщо результат - список, беремо перший елемент
            if isinstance(predictions, list):
                predictions = predictions[0]

            emotions = predictions["emotion"]

            # Перелік усіх потрібних емоцій
            all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']

            # Створюємо вектор з усіма емоціями, вставляючи 0 для відсутніх
            emotion_vector = [emotions.get(emotion, 0.0) for emotion in all_emotions]
            emotion_vector = np.array(emotion_vector, dtype=np.float32)

            # Нормалізація, щоб сума ймовірностей дорівнювала 1
            emotion_vector = emotion_vector / np.sum(emotion_vector)

            return emotion_vector
        except Exception as e:
            print(f"Помилка аналізу емоцій: {e}")
            # Повертаємо дефолтний вектор емоцій у разі помилки
            default_vector = np.array([0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.6, 0.05], dtype=np.float32)
            return default_vector
    except Exception as e:
        print(f"Критична помилка при обробці зображення: {e}")
        raise HTTPException(status_code=400, detail=f"Помилка при аналізі зображення: {str(e)}")

def calibrated_predict_with_model(emotion_vector):
    """
    Виконує прогнозування рівня стресу на основі емоційного вектора.

    Args:
        emotion_vector: Вектор емоцій (список або numpy array).

    Returns:
        predicted_class (str): Передбачений рівень стресу ('Low', 'Middle', 'High').
        probabilities (dict): Ймовірності для кожного рівня стресу.
    """
    global model_name, stress_model, model_metadata, feature_importance, learned_weights
    
    if stress_model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    stress_model.eval()
    input_tensor = torch.tensor(emotion_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = stress_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Визначення рівня стресу за найбільшою ймовірністю
    stress_levels = ['Low', 'Middle', 'High']
    predicted_index = np.argmax(probabilities)
    predicted_class = stress_levels[predicted_index]
    
    # Отримання метрик для цього класу стресу
    class_metrics = {}
    if model_metadata and 'metrics' in model_metadata:
        metrics = model_metadata['metrics']
        for metric_name in ['precision', 'recall', 'f1_score']:
            if metric_name in metrics and predicted_class in metrics[metric_name]:
                class_metrics[metric_name] = metrics[metric_name][predicted_class]
    
    # Перетворення ймовірностей у словник
    prob_dict = {level: float(prob) for level, prob in zip(stress_levels, probabilities)}

    return predicted_class, prob_dict, float(probabilities[predicted_index]), class_metrics

class StressResponse(BaseModel):
    stress_level: str
    probabilities: dict
    confidence: float
    metrics: Optional[dict] = None

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Тут виклик функції аналізу стресу
        print("Викликаємо функцію аналізу стресу...")
        
        global model_name, stress_model, model_metadata, feature_importance, learned_weights
        
        print(f"Отримано запит на аналіз зображення: {file.filename}")
        
        # Перевірте, чи модель стресу завантажена
        if stress_model is None:
            print("Завантаження моделі стресу...")
            stress_model, feature_importance, learned_weights, model_metadata = load_weighted_model(model_name)
        
        # Зчитуємо зображення
        contents = await file.read()
        print(f"Зчитано {len(contents)} байтів даних")
        
        # Декодуємо зображення
        image = Image.open(io.BytesIO(contents))
        print(f"Зображення успішно декодовано, розмір: {image.size}")
        
        # Перетворюємо на массив numpy
        img_array = np.array(image)
        print(f"Зображення перетворено на масив numpy, форма: {img_array.shape}")
        
        # Тут виклик функції аналізу емоцій
        print("Викликаємо функцію аналізу емоцій...")
        emotions_result = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
        
        # Отримання емоцій у потрібному форматі для моделі стресу
        emotions = {}
        if isinstance(emotions_result, list):
            emotions = emotions_result[0]['emotion']
        else:
            emotions = emotions_result['emotion']
        
        print(f"Результат аналізу емоцій: {emotions}")
        
        # Підготовка вхідних даних для моделі стресу
        input_features = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt"]
        input_values = []
        for feature in input_features:
            if feature == "contempt" and "contempt" not in emotions:
                # Якщо contempt відсутній у результатах DeepFace, встановлюємо значення 0
                input_values.append(0)
            else:
                input_values.append(emotions.get(feature, 0))
        
        # Тут виклик функції аналізу стресу
        print("Викликаємо функцію аналізу стресу...")
        
        # Перевірте, чи модель стресу завантажена
        if stress_model is None:
            stress_model, _, _, _ = load_weighted_model("stress_detector_weighted_nn")
            
        # Перетворення на тензор
        input_tensor = torch.tensor([input_values], dtype=torch.float32)
        
        # Прогнозування
        with torch.no_grad():
            stress_output = stress_model(input_tensor)
            
        # Перетворення результату
        stress_probabilities = torch.softmax(stress_output, dim=1).numpy()[0]
        stress_classes = ["Low", "Middle", "High"]
        stress_result = {stress_classes[i]: float(stress_probabilities[i]) for i in range(len(stress_classes))}
        
        # Визначення найбільш ймовірного класу стресу
        predicted_stress = stress_classes[np.argmax(stress_probabilities)]
        
        # Формування результату
        result = {
            "emotions": emotions,
            "stress_probabilities": stress_result,
            "predicted_stress": predicted_stress
        }
        
        print(f"Результат аналізу стресу: {result}")
        
        # Результат
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"Помилка при аналізі зображення: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "detail": str(e)}, 500

@app.post("/predict/base64", response_model=StressResponse)
async def predict_from_base64(base64_image: str = Form(...)):
    """
    Прогнозування рівня стресу на основі зображення у форматі base64.
    """
    global model_name, stress_model, model_metadata, feature_importance, learned_weights
    try:
        # Декодування base64 у бінарні дані
        image_data = base64.b64decode(base64_image)
        emotion_vector = generate_emotion_vector(image_data)
        
        stress_level, probabilities, confidence = calibrated_predict_with_model(emotion_vector)
        
        return {
            "stress_level": stress_level,
            "probabilities": probabilities,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/emotions", response_model=StressResponse)
async def predict_from_emotions(emotion_data: EmotionVector):
    """
    Прогнозування рівня стресу на основі вектора емоцій.
    """
    try:
        global model_name, stress_model, model_metadata, feature_importance, learned_weights
        # Перевірити, чи модель завантажена
        if stress_model is None:
            print("Модель не завантажена, спроба завантажити...")
            stress_model, feature_importance, learned_weights, model_metadata = load_weighted_model(model_name)
            if stress_model is None:
                raise HTTPException(status_code=500, detail="Не вдалося завантажити модель стресу")
            
        emotions = np.array(emotion_data.emotions, dtype=np.float32)
        
        if len(emotions) != 8:
            raise HTTPException(
                status_code=400, 
                detail="Вектор емоцій повинен містити 8 значень (angry, disgust, fear, happy, sad, surprise, neutral, contempt)"
            )
        
        # Нормалізація вектора емоцій
        if np.sum(emotions) > 0:
            emotions = emotions / np.sum(emotions)
        
        stress_level, probabilities, confidence = calibrated_predict_with_model(emotions)
        
        return {
            "stress_level": stress_level,
            "probabilities": probabilities,
            "confidence": confidence
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck")
async def healthcheck():
    """
    Перевірка стану API.
    """
    global model_name, stress_model, model_metadata
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model_info")
async def get_model_info():
    try:
        global model_name, stress_model, model_metadata
        
        print("Спроба отримати інформацію про модель...")
        
        # Спершу перевірте, чи модель вже завантажена
        if stress_model is None:
            # Завантажте модель, якщо вона ще не завантажена
            stress_model, _, _, model_metadata = load_weighted_model(model_name)
        
        # Створіть об'єкт model_info
        model_info = {
            "model_type": model_metadata.get("model_type", "StressDetectionNN"),
            "input_features": model_metadata.get("input_features", []),
            "metrics": model_metadata.get("metrics", {}),
            "date_saved": model_metadata.get("date_saved", "N/A")
        }
        
        return {"status": "success", "model_info": model_info}
    except Exception as e:
        print(f"Помилка при отриманні інформації про модель: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Повертаємо помилку з кодом 500, а не 200
        return {"status": "error", "message": str(e)}, 500

@app.get("/model_metrics")
async def model_metrics():
    """
    Надає детальні метрики продуктивності моделі.
    """
    global stress_model, model_metadata
    if stress_model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    if not model_metadata or 'metrics' not in model_metadata:
        raise HTTPException(status_code=404, detail="Метрики моделі не знайдено")
    
    return {
        "model_name": model_name,
        "metrics": model_metadata.get('metrics', {}),
        "date_saved": model_metadata.get('date_saved', 'невідома')
    }

@app.get("/model_performance/roc_curve")
async def roc_curve():
    """
    Повертає дані для побудови ROC кривої моделі.
    """
    global stress_model, model_metadata
    if stress_model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    # В реальному випадку тут би ви завантажували дані для ROC кривої
    # Для демонстрації створюємо приблизні дані
    
    # ROC крива для кожного класу
    roc_data = {
        "Low": {
            "fpr": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "tpr": [0.0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0],
            "auc": 0.82
        },
        "Middle": {
            "fpr": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "tpr": [0.0, 0.3, 0.5, 0.65, 0.75, 0.8, 0.85, 0.87, 0.9, 0.95, 0.98, 1.0],
            "auc": 0.78
        },
        "High": {
            "fpr": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "tpr": [0.0, 0.35, 0.55, 0.7, 0.78, 0.82, 0.87, 0.9, 0.93, 0.96, 0.98, 1.0],
            "auc": 0.81
        }
    }
    
    return roc_data

if __name__ == "__main__":
    import uvicorn
    
    # Визначення порту з env змінної (для render.com) або використання 8000 за замовчуванням
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)