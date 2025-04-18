import os
import torch
import numpy as np
import io
from PIL import Image
from io import BytesIO
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_307_TEMPORARY_REDIRECT
from pydantic import BaseModel
import cv2
from deepface import DeepFace


# improt special modules
from model_loader import load_weighted_model

# Global values
stress_model = None
model_metadata = None
feature_importance = None
learned_weights = None

app = FastAPI(title="Stress Detection API", description="Facial expression stress recognition API.")

# setting CORS for web clien aссess
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model while server is starting
model_name = "stress_detector_weighted_nn"

# if there are static files, connect
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    """
    returns speedometer.html when goes to root "/"
    """
    return FileResponse("static/speedometer.html")

@app.get("/speedometer", response_class=HTMLResponse)
async def serve_speedometer():
    """
    Returns speedometer.html
    """
    return FileResponse("static/speedometer.html")

def default_html_response():
    """
    returns default HTML in case, if file not found
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

# check if Deepface works properly
try:
    # Create dummy image for test
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_result = DeepFace.analyze(img_path=dummy_img, actions=['emotion'], enforce_detection=False)
    print("DeepFace успішно ініціалізовано")
except Exception as e:
    print(f"Error while initialize DeepFace: {str(e)}")
    import traceback
    print(traceback.format_exc())

@app.on_event("startup")
async def startup_event():
    global model_name, stress_model, model_metadata, feature_importance, learned_weights
    try:
        print("loading model while starting...")
        stress_model, feature_importance, learned_weights, model_metadata = load_weighted_model(model_name)
        print("Model loaded successfuly")
    except Exception as e:
        print(f"Error while loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("Checking DeepFace...")
    try:
        import os
        #Check model path
        model_path = os.path.join(os.path.expanduser('~'), '.deepface', 'weights', 'facial_expression_model_weights.h5')
        if os.path.isfile(model_path):
            print(f"Model DeepFace was found by path {model_path}")
        else:
            print(f"Model DeepFace was not found by path {model_path}")
            # Alternative way
            if os.path.isfile("/root/.deepface/weights/efacial_expression_model_weights.h5"):
                print("Model was found by an alternative path")
            else:
                print("Model was not found by an alternative path")
    except Exception as e:
        print(f"Error while checking DeepFace: {e}")

class EmotionVector(BaseModel):
    emotions: List[float]

class StressResponse(BaseModel):
    stress_level: str
    probabilities: dict
    confidence: float

def generate_emotion_vector(image_data):
    """
    Generate emotions vector with DeepFace model.
    """
    try:
        # Converts bytes array into image
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

            # If list, take firs element
            if isinstance(predictions, list):
                predictions = predictions[0]

            emotions = predictions["emotion"]

            # list of all necessary emotions
            all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']

            # Create vector with all emotions, paste 0 for missing
            emotion_vector = [emotions.get(emotion, 0.0) for emotion in all_emotions]
            emotion_vector = np.array(emotion_vector, dtype=np.float32)

            # noramlise, make the sum of probabilities equal to 1
            emotion_vector = emotion_vector / np.sum(emotion_vector)

            return emotion_vector
        except Exception as e:
            print(f"Помилка аналізу емоцій: {e}")
            # Return default vector of emotins in error case
            default_vector = np.array([0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.6, 0.05], dtype=np.float32)
            return default_vector
    except Exception as e:
        print(f"Critical error while proccesing image: {e}")
        raise HTTPException(status_code=400, detail=f"error while proccesing image: {str(e)}")

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

# Додайте цю функцію для зменшення розміру зображення перед обробкою
def resize_image_if_needed(image, max_size=800):
    """
    Зменшує розмір зображення, якщо воно завелике.
    Це допомагає запобігти помилкам OOM (Out of Memory) на сервері.
    """
    width, height = image.size
    
    # Перевіряємо, чи зображення завелике
    if width > max_size or height > max_size:
        # Обчислюємо новий розмір зі збереженням пропорцій
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Змінюємо розмір зображення
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    
    return image



@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Додаємо інформацію про запит у логи
        print(f"Отримано запит на аналіз зображення: {file.filename}, content_type: {file.content_type}")
        
        global model_name, stress_model, model_metadata, feature_importance, learned_weights
        
        # Перевірте, чи модель стресу завантажена
        if stress_model is None:
            print("Завантаження моделі стресу...")
            stress_model, feature_importance, learned_weights, model_metadata = load_weighted_model(model_name)
        
        # Зчитуємо зображення
        contents = await file.read()
        file_size = len(contents)
        print(f"Зчитано {file_size} байтів даних")
        
        # Встановлюємо обмеження на розмір файлу (20 МБ)
        max_file_size = 20 * 1024 * 1024  # 20 МБ у байтах
        if file_size > max_file_size:
            return {"status": "error", "detail": "Файл завеликий. Максимальний розмір - 20 МБ"}
        
        # Декодуємо зображення
        try:
            image = Image.open(io.BytesIO(contents))
            print(f"Зображення успішно декодовано, розмір: {image.size}, формат: {image.format}")
            
            # Зменшуємо розмір зображення, якщо воно завелике
            max_image_size = 800  # Максимальний розмір сторони зображення
            image = resize_image_if_needed(image, max_image_size)
            print(f"Розмір зображення після обробки: {image.size}")
            
            # Виправлення орієнтації для мобільних фото
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                
                exif = image._getexif()
                if exif is not None and orientation in exif:
                    if exif[orientation] == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif exif[orientation] == 3:
                        image = image.transpose(Image.ROTATE_180)
                    elif exif[orientation] == 4:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif exif[orientation] == 5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                    elif exif[orientation] == 6:
                        image = image.transpose(Image.ROTATE_270)
                    elif exif[orientation] == 7:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                    elif exif[orientation] == 8:
                        image = image.transpose(Image.ROTATE_90)
                
                print("Орієнтацію зображення перевірено та виправлено за потреби")
            except Exception as e:
                print(f"Помилка при виправленні орієнтації: {str(e)}")
            
            # Конвертуємо в RGB, якщо потрібно
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Зображення конвертовано в RGB режим")
            
            # Перетворюємо на масив numpy
            img_array = np.array(image)
            print(f"Зображення перетворено на масив numpy, форма: {img_array.shape}")
            
            # Тут виклик функції аналізу емоцій
            print("Викликаємо функцію аналізу емоцій...")
            
            # Обробляємо виняткові ситуації з DeepFace
            try:
                emotions_result = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
                print(f"DeepFace успішно проаналізував зображення")
            except Exception as deepface_error:
                print(f"Помилка при першій спробі аналізу DeepFace: {str(deepface_error)}")
                
                # Спробуємо покращити зображення та повторити аналіз
                try:
                    # Перетворюємо в сірий колір і нормалізуємо
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    normalized = cv2.equalizeHist(gray)
                    enhanced = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                    
                    print("Зображення покращено для розпізнавання обличчя")
                    emotions_result = DeepFace.analyze(img_path=enhanced, actions=['emotion'], enforce_detection=False)
                    print("DeepFace успішно проаналізував покращене зображення")
                except Exception as enhanced_error:
                    print(f"Помилка при аналізі покращеного зображення: {str(enhanced_error)}")
                    return {"status": "error", "detail": "Не вдалося виявити обличчя на зображенні. Будь ласка, спробуйте інше фото."}
            
            # Отримання емоцій у потрібному форматі для моделі стресу
            emotions = {}
            if isinstance(emotions_result, list):
                if len(emotions_result) == 0:
                    return {"status": "error", "detail": "Обличчя не знайдено на зображенні"}
                emotions = emotions_result[0]['emotion']
            else:
                emotions = emotions_result['emotion']
            
            print(f"Результат аналізу емоцій: {emotions}")
            
            # Підготовка вхідних даних для моделі стресу
            input_features = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt"]
            input_values = []
            for feature in input_features:
                if feature == "contempt" and "contempt" not in emotions:
                    input_values.append(0)
                else:
                    input_values.append(emotions.get(feature, 0))
            
            # Нормалізуємо значення
            sum_values = sum(input_values)
            if sum_values > 0:
                input_values = [v / sum_values for v in input_values]
            
            # Прогнозування стресу
            print("Викликаємо функцію аналізу стресу...")
            
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
            
            print(f"Результат аналізу стресу: {predicted_stress}")
            
            # Результат
            return {"status": "success", "result": result}
            
        except Exception as image_error:
            print(f"Помилка при обробці зображення: {str(image_error)}")
            return {"status": "error", "detail": f"Помилка при обробці зображення: {str(image_error)}"}
            
    except Exception as e:
        print(f"Загальна помилка при аналізі зображення: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "detail": str(e)}


@app.get("/healthcheck")
async def healthcheck():
    """
    Перевірка стану API.
    """
    global model_name, stress_model, model_metadata
    return {"status": "ok", "model_loaded": stress_model is not None}
