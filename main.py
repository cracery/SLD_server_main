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


from model_loader import load_weighted_model


stress_model = None
model_metadata = None
ftr_importance = None
learned_wghts = None

app = FastAPI(title="Stress Detection API", description="Facial expression stress recognition API.")

# setting CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_name = "stress_detector_weighted_nn"

# connect static
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    """
    returns sld.html when goes to root "/"
    """
    return FileResponse("static/sld.html")

@app.get("/sld", response_class=HTMLResponse)
async def serve_speedometer():
    return FileResponse("static/sld.html")

def default_html_response():
    """
    returns default, if file not found
    """
    return FileResponse("static/Default.html")



# check if Deepface works properly
try:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_result = DeepFace.analyze(img_path=dummy_img, actions=['emotion'], enforce_detection=False)
    print("DeepFace initialised")
except Exception as e:
    print(f"Error while initialize DeepFace: {str(e)}")
    import traceback
    print(traceback.format_exc())

@app.on_event("startup")
async def startup_event():
    global model_name, stress_model, model_metadata, ftr_importance, learned_wghts
    try:
        print("loading model while starting...")
        stress_model, ftr_importance, learned_wghts, model_metadata = load_weighted_model(model_name)
        print("Model loaded successfuly")
    except Exception as e:
        print(f"Error while loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("Checking DeepFace...")
    try:
        import os


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

class Emot_Vector(BaseModel):
    emotions: List[float]

class Stress_respo(BaseModel):
    stress_level: str
    probabilities: dict
    confidence: float

def generate_emotion_vector(img_data):
    """
    Generate emotions vector with DeepFace model.
    """
    try:
        # Converts bytes array into image
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

            if isinstance(predictions, list):
                predictions = predictions[0]

            emotions = predictions["emotion"]


            all_emot = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']

            # Create vector with all emotions
            emot_vector = [emotions.get(emotion, 0.0) for emotion in all_emot]
            emot_vector = np.array(emot_vector, dtype=np.float32)

            # noramlise
            emot_vector = emot_vector / np.sum(emot_vector)

            return emot_vector
        

        except Exception as e:
            print(f"Помилка аналізу емоцій: {e}")
            default_vector = np.array([0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.6, 0.05], dtype=np.float32)
            return default_vector
    except Exception as e:
        print(f"Critical error while proccesing image: {e}")
        raise HTTPException(status_code=400, detail=f"error while proccesing image: {str(e)}")

def calibrated_predict_with_model(emotion_vector):
    """
    Make stress level prediction besed on emotions vector.(lvls + probs)
    """
    global model_name, stress_model, model_metadata, ftr_importance, learned_wghts
    
    if stress_model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    stress_model.eval()
    input_tensor = torch.tensor(emotion_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = stress_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Choose stress level with the biggest posibility
    stress_levels = ['Low', 'Middle', 'High']
    pred_index = np.argmax(probs)
    pred_class = stress_levels[pred_index]
    
    # Get stress class metrics
    class_metrics = {}
    if model_metadata and 'metrics' in model_metadata:
        metrics = model_metadata['metrics']
        for metric_name in ['precision', 'recall', 'f1_score']:
            if metric_name in metrics and pred_class in metrics[metric_name]:
                class_metrics[metric_name] = metrics[metric_name][pred_class]
    

    prob_dict = {level: float(prob) for level, prob in zip(stress_levels, probs)}
    return pred_class, prob_dict, float(probs[pred_index]), class_metrics

class Stress_respo(BaseModel):
    stress_level: str
    probs: dict
    confidence: float
    metrics: Optional[dict] = None


# Preprocess iamge
def resize_img(img, max_size=800):
    wdth, height = img.size
    
    if wdth > max_size or height > max_size:
        if wdth > height:
            new_wdth = max_size
            new_height = int(height * (max_size / wdth))
        else:
            new_height = max_size
            new_wdth = int(wdth * (max_size / height))
        

        resized_img = img.resize((new_wdth, new_height), Image.LANCZOS)
        return resized_img
    
    return img



@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Add request info into logs
        print(f"Request was got: {file.filename}, content_type: {file.content_type}")
        
        global model_name, stress_model, model_metadata, ftr_importance, learned_wghts
        
        if stress_model is None:
            print("Loading stress model...")
            stress_model, ftr_importance, learned_wghts, model_metadata = load_weighted_model(model_name)
        
        # Read img
        contents = await file.read()
        file_size = len(contents)
        print(f"Read {file_size} data bytes")
        

        max_file_size = 20 * 1024 * 1024
        if file_size > max_file_size:
            return {"status": "error", "detail": "File is too big. Max size - 20 mb"}
        
        # Decode image
        try:
            img = Image.open(io.BytesIO(contents))
            print(f"Image successfully decoded, size: {img.size}, format: {img.format}")


            max_image_size = 800 
            img = resize_img(img, max_image_size)
            print(f"Image size after processing: {img.size}")
            

            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"Image converted into RGB")
            
            # Convert into numpy array
            img_array = np.array(img)
            print(f"Image convert into numpy array, form: {img_array.shape}")
            
            print("Call emotion analysis function...")
            
            # Process exceptional cases
            try:
                emots_result = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
                print(f"DeepFace process image successfuly")
            except Exception as deepface_error:
                print(f"Error on the first try to analyse DeepFace: {str(deepface_error)}")
                
                try:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    normalized = cv2.equalizeHist(gray)
                    enhanced = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                    
                    print("Image enhanced for face recognition")
                    emots_result = DeepFace.analyze(img_path=enhanced, actions=['emotion'], enforce_detection=False)
                    print("DeepFace successfully analyses enhanced image")
                except Exception as enhanced_error:
                    print(f"Error when analysing an enhanced image: {str(enhanced_error)}")
                    return {"status": "error", "detail": "The face in the image could not be detected. Please try another photo."}
            
            # Get emotions
            emots = {}
            if isinstance(emots_result, list):
                if len(emots_result) == 0:
                    return {"status": "error", "detail": "Face was not found"}
                emots = emots_result[0]['emotion']
            else:
                emots = emots_result['emotion']
            
            print(f"Emotion analysis result: {emots}")
            
            # Prepare input
            input_ftrs = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt"]
            input_values = []
            for ftr in input_ftrs:
                if ftr == "contempt" and "contempt" not in emots:
                    input_values.append(0)
                else:
                    input_values.append(emots.get(ftr, 0))
            
            # Normalise
            sum_values = sum(input_values)
            if sum_values > 0:
                input_values = [v / sum_values for v in input_values]
            

            print("Call the stress analysis function...")
            

            input_tensor = torch.tensor([input_values], dtype=torch.float32)
            
            with torch.no_grad():
                stress_output = stress_model(input_tensor)
                
            stress_probs = torch.softmax(stress_output, dim=1).numpy()[0]
            stress_classes = ["Low", "Middle", "High"]
            stress_result = {stress_classes[i]: float(stress_probs[i]) for i in range(len(stress_classes))}
            

            predicted_stress = stress_classes[np.argmax(stress_probs)]
            

            result = {
                "emotions": emots,
                "stress_probabilities": stress_result,
                "predicted_stress": predicted_stress
            }
            
            print(f"Stress analysis result: {predicted_stress}")
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
    global model_name, stress_model, model_metadata
    return {"status": "ok", "model_loaded": stress_model is not None}
