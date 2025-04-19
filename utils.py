import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from io import BytesIO
from PIL import Image

def ensure_dir_exists(dir_path: str) -> None:
    """
    Checks the existence of directory and creates it if need.
    Args:
        dir_path: Directory path
    """
    os.makedirs(dir_path, exist_ok=True)

def preprocess_image_for_deepface(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for DeepFace.
    Args:
        img: Images in numpy array format
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Convert in RGB
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def format_emotion_vector(emotions: Dict[str, float]) -> np.ndarray:
    """
    Format emotion dictionary into a normalised vector.
    
    Args:
        emotions: Dictionary with emotion probabilities
        
    Returns:
        np.ndarray: Normalised vector of emotions
    """
    #List all emotions in the correct order
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    
    # Create emotions vector
    emotion_vector = np.array([emotions.get(emotion, 0.0) for emotion in all_emotions], dtype=np.float32)
    
    # Normalise
    if np.sum(emotion_vector) > 0:
        emotion_vector = emotion_vector / np.sum(emotion_vector)
    
    return emotion_vector

def format_stress_result(level: str, probabilities: Dict[str, float], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Format results of stress prediction.
    
    Args:
        level: Stress level ('Low', 'Middle', 'High')
        probabilities: Dictionary with probabilities for each level
        threshold: Threshold for determining assurance
        
    Returns:
        dict: Formated results
    """
    # Find highest propability
    max_prob = max(probabilities.values())
    
    # assurance message
    if max_prob >= threshold:
        confidence_message = "High assurance"
    else:
        confidence_message = "Low assurance"
    
    # Round to 2 symbols
    formatted_probs = {k: round(v * 100, 2) for k, v in probabilities.items()}
    
    return {
        "stress_level": level,
        "probabilities": formatted_probs,
        "confidence": round(max_prob * 100, 2),
        "message": f"Stress level detected '{level}' з {confidence_message} ({round(max_prob * 100, 2)}%)"
    }
