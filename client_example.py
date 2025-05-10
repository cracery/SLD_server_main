import requests
import json
import argparse
from pathlib import Path
import pprint

def predict_from_img(api_url, image_path):
    """
    Send request for prediction of stress levels by image
    """
    if not Path(image_path).exists():
        print(f"Error: File {image_path} doesn't exist")
        return None
    files= {
        'file': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')
    }
    try:
        respo= requests.post(f"{api_url}/predict/image", files=files)
        files['file'][1].close()
        
        if respo.status_code== 200:
            return respo.json()
        else:
            print(f"error: {respo.status_code} - {respo.text}")
            return None
    except Exception as e:
        print(f"error while sending request: {str(e)}")
        return None
