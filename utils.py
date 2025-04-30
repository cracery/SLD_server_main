import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from io import BytesIO
from PIL import Image

def ensure_dir_exists(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)

def preprocess_image_for_deepface(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for DeepFace.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def format_emotion_vector(emotions: Dict[str, float]) -> np.ndarray:
    """
    Format emotion scores into a normalised vector.
    """
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    emo_vec = np.array([emotions.get(emotion, 0.0) for emotion in all_emotions], dtype=np.float32)
    
    if np.sum(emo_vec) > 0:
        emo_vec = emo_vec / np.sum(emo_vec)
    
    return emo_vec

def format_stress_result(level: str, probabilities: Dict[str, float], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Format results of stress prediction.
    """
    max_prob = max(probabilities.values())
    
    # assurance
    if max_prob >= threshold:
        confidence_message = "High assurance"
    else:
        confidence_message = "Low assurance"
    formatted_probs = {k: round(v * 100, 2) for k, v in probabilities.items()}
    
    return {
        "stress_level": level,
        "probabilities": formatted_probs,
        "confidence": round(max_prob * 100, 2),
        "message": f"Stress level detected '{level}' with {confidence_message} ({round(max_prob * 100, 2)}%)"
    }
