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
    returns Default.html in case, if file not found
    """
    return FileResponse("static/Default.html")



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
    Make stress level prediction besed on emotions vector.

    Args:
        emotion_vector: emotions vector (list or numpy array).

    Returns:
        predicted_class (str): Predicted stress level ('Low', 'Middle', 'High').
        probabilities (dict): Posibilities for every stress level.
    """
    global model_name, stress_model, model_metadata, feature_importance, learned_weights
    
    if stress_model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    stress_model.eval()
    input_tensor = torch.tensor(emotion_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = stress_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Chose stress level with the biggest posibility
    stress_levels = ['Low', 'Middle', 'High']
    predicted_index = np.argmax(probabilities)
    predicted_class = stress_levels[predicted_index]
    
    # Get metrics for this stress class
    class_metrics = {}
    if model_metadata and 'metrics' in model_metadata:
        metrics = model_metadata['metrics']
        for metric_name in ['precision', 'recall', 'f1_score']:
            if metric_name in metrics and predicted_class in metrics[metric_name]:
                class_metrics[metric_name] = metrics[metric_name][predicted_class]
    
    # Convert posibilities into dictionary
    prob_dict = {level: float(prob) for level, prob in zip(stress_levels, probabilities)}

    return predicted_class, prob_dict, float(probabilities[predicted_index]), class_metrics

class StressResponse(BaseModel):
    stress_level: str
    probabilities: dict
    confidence: float
    metrics: Optional[dict] = None

# Make picture smaller before proceed
def resize_image_if_needed(image, max_size=800):
    """
    Make picure smaller to avoid OOM (Out of Memory) errors on server.
    """
    width, height = image.size
    
    # Check if picture too big
    if width > max_size or height > max_size:
        # Calculate new size with saving properties
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Resize
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    
    return image



@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Add request info into logs
        print(f"Request was got: {file.filename}, content_type: {file.content_type}")
        
        global model_name, stress_model, model_metadata, feature_importance, learned_weights
        
        # Check if model loaded
        if stress_model is None:
            print("Loading stress model...")
            stress_model, feature_importance, learned_weights, model_metadata = load_weighted_model(model_name)
        
        # Read image
        contents = await file.read()
        file_size = len(contents)
        print(f"Read {file_size} data bytes")
        
        # Set file size limits (20 mb)
        max_file_size = 20 * 1024 * 1024
        if file_size > max_file_size:
            return {"status": "error", "detail": "File is too big. Max size - 20 mb"}
        
        # Decode image
        try:
            image = Image.open(io.BytesIO(contents))
            print(f"Зображення успішно декодовано, розмір: {image.size}, формат: {image.format}")

            # Make image size lower
            max_image_size = 800 
            image = resize_image_if_needed(image, max_image_size)
            print(f"Image size after processing: {image.size}")
            
            # Convert into RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Image converted into RGB")
            
            # Convert into numpy array
            img_array = np.array(image)
            print(f"Image convert into numpy array, form: {img_array.shape}")
            
            print("Call emotion analysis function...")
            
            # Process exceptional situations with DeepFace
            try:
                emotions_result = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
                print(f"DeepFace process image successfuly")
            except Exception as deepface_error:
                print(f"Error on the first try to analyse DeepFace: {str(deepface_error)}")
                
                # Try to enchance and rerty analysis
                try:
                    # Convert into grayscale and normalise
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    normalized = cv2.equalizeHist(gray)
                    enhanced = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                    
                    print("Image enhanced for face recognition")
                    emotions_result = DeepFace.analyze(img_path=enhanced, actions=['emotion'], enforce_detection=False)
                    print("DeepFace successfully analyses enhanced image")
                except Exception as enhanced_error:
                    print(f"Error when analysing an enhanced image: {str(enhanced_error)}")
                    return {"status": "error", "detail": "The face in the image could not be detected. Please try another photo."}
            
            # Get emotions in the correct format for the stress model
            emotions = {}
            if isinstance(emotions_result, list):
                if len(emotions_result) == 0:
                    return {"status": "error", "detail": "Face was not found"}
                emotions = emotions_result[0]['emotion']
            else:
                emotions = emotions_result['emotion']
            
            print(f"Emotion analysis result: {emotions}")
            
            # Prepare input data for the stress model
            input_features = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt"]
            input_values = []
            for feature in input_features:
                if feature == "contempt" and "contempt" not in emotions:
                    input_values.append(0)
                else:
                    input_values.append(emotions.get(feature, 0))
            
            # Normalise value
            sum_values = sum(input_values)
            if sum_values > 0:
                input_values = [v / sum_values for v in input_values]
            
            # Predict stress
            print("Call the stress analysis function...")
            
            # Convert to a tensor
            input_tensor = torch.tensor([input_values], dtype=torch.float32)
            
            # Predict
            with torch.no_grad():
                stress_output = stress_model(input_tensor)
                
            # Convert results
            stress_probabilities = torch.softmax(stress_output, dim=1).numpy()[0]
            stress_classes = ["Low", "Middle", "High"]
            stress_result = {stress_classes[i]: float(stress_probabilities[i]) for i in range(len(stress_classes))}
            
            # Determining the most likely stress class
            predicted_stress = stress_classes[np.argmax(stress_probabilities)]
            
            # Form result
            result = {
                "emotions": emotions,
                "stress_probabilities": stress_result,
                "predicted_stress": predicted_stress
            }
            
            print(f"Stress analysis result: {predicted_stress}")
            
            # Result
            return {"status": "success", "result": result}
            
        except Exception as image_error:
            print(f"Image processing error: {str(image_error)}")
            return {"status": "error", "detail": f"Image processing error: {str(image_error)}"}
            
    except Exception as e:
        print(f"General image analysis error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "detail": str(e)}


@app.get("/healthcheck")
async def healthcheck():
    """
    API state check.
    """
    global model_name, stress_model, model_metadata
    return {"status": "ok", "model_loaded": stress_model is not None}
