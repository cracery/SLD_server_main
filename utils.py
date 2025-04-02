import os
import base64
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from io import BytesIO
from PIL import Image

def ensure_dir_exists(dir_path: str) -> None:
    """
    Перевіряє існування директорії і створює її за необхідністю.
    
    Args:
        dir_path: Шлях до директорії
    """
    os.makedirs(dir_path, exist_ok=True)

def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Декодує base64-кодоване зображення у масив numpy.
    
    Args:
        base64_str: Base64-кодоване зображення
        
    Returns:
        np.ndarray: Декодоване зображення у форматі numpy array
    """
    try:
        # Видалення префікса base64 якщо він є
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
            
        # Декодування base64
        image_data = base64.b64decode(base64_str)
        
        # Конвертація у numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Декодування зображення за допомогою OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Конвертація BGR в RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    except Exception as e:
        raise ValueError(f"Помилка при декодуванні base64-зображення: {str(e)}")

def save_image_from_base64(base64_str: str, output_path: str) -> str:
    """
    Зберігає base64-кодоване зображення як файл.
    
    Args:
        base64_str: Base64-кодоване зображення
        output_path: Шлях для збереження зображення
        
    Returns:
        str: Повний шлях до збереженого зображення
    """
    try:
        # Декодування зображення
        img = decode_base64_image(base64_str)
        
        # Перевірка директорії
        output_dir = os.path.dirname(output_path)
        ensure_dir_exists(output_dir)
        
        # Збереження зображення
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return output_path
    except Exception as e:
        raise ValueError(f"Помилка при збереженні зображення: {str(e)}")

def preprocess_image_for_deepface(img: np.ndarray) -> np.ndarray:
    """
    Препроцесінг зображення для DeepFace.
    
    Args:
        img: Зображення у форматі numpy array
        
    Returns:
        np.ndarray: Препроцесоване зображення
    """
    # Конвертація у RGB (якщо необхідно)
    if len(img.shape) == 2:  # Зображення у відтінках сірого
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Зображення з альфа-каналом
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img

def format_emotion_vector(emotions: Dict[str, float]) -> np.ndarray:
    """
    Форматує словник емоцій у нормалізований вектор.
    
    Args:
        emotions: Словник з ймовірностями емоцій
        
    Returns:
        np.ndarray: Нормалізований вектор емоцій
    """
    # Перелік усіх емоцій у потрібному порядку
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    
    # Створюємо вектор емоцій
    emotion_vector = np.array([emotions.get(emotion, 0.0) for emotion in all_emotions], dtype=np.float32)
    
    # Нормалізація вектора
    if np.sum(emotion_vector) > 0:
        emotion_vector = emotion_vector / np.sum(emotion_vector)
    
    return emotion_vector

def format_stress_result(level: str, probabilities: Dict[str, float], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Форматує результати прогнозування стресу.
    
    Args:
        level: Рівень стресу ('Low', 'Middle', 'High')
        probabilities: Словник з ймовірностями для кожного рівня
        threshold: Поріг для визначення впевненості
        
    Returns:
        dict: Форматовані результати
    """
    # Знаходимо найбільшу ймовірність
    max_prob = max(probabilities.values())
    
    # Визначаємо повідомлення щодо впевненості
    if max_prob >= threshold:
        confidence_message = "високою впевненістю"
    else:
        confidence_message = "низькою впевненістю"
    
    # Форматування ймовірностей (округлення до 2 знаків після коми)
    formatted_probs = {k: round(v * 100, 2) for k, v in probabilities.items()}
    
    return {
        "stress_level": level,
        "probabilities": formatted_probs,
        "confidence": round(max_prob * 100, 2),
        "message": f"Виявлено рівень стресу '{level}' з {confidence_message} ({round(max_prob * 100, 2)}%)"
    }