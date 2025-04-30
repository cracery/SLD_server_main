import requests
import json
import argparse
from pathlib import Path
import pprint

def predict_from_image(api_url, image_path):
    """
    Send request for prediction of stress levels by image
    """
    if not Path(image_path).exists():
        print(f"Error: File {image_path} doesn't exist")
        return None
    files = {
        'file': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')
    }
    try:
        response = requests.post(f"{api_url}/predict/image", files=files)
        files['file'][1].close()
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"error while sending request: {str(e)}")
        return None
