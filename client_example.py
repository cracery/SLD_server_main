import requests
import json
import argparse
from pathlib import Path
import pprint

def predict_from_image(api_url, image_path):
    """
    Відправляє запит на прогнозування рівня стресу за зображенням
    
    Args:
        api_url: URL API server
        image_path: image path
    
    Returns:
        dictionary with prediction results
    """
    # check if file exists
    if not Path(image_path).exists():
        print(f"Помилка: Файл {image_path} не існує")
        return None
    
    # prepare sending file
    files = {
        'file': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        # send request
        response = requests.post(f"{api_url}/predict/image", files=files)
        
        # close file
        files['file'][1].close()
        
        # process answer
        if response.status_code == 200:
            return response.json()
        else:
            print(f"error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"error while sending request: {str(e)}")
        return None
"""
parser = argparse.ArgumentParser(description='Stress recognition client API')
parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                    help='URL server API (default: http://localhost:8000)')

subparsers = parser.add_subparsers(dest='command', help='Команда')

# image parser
image_parser = subparsers.add_parser('image', help='Прогнозування за зображенням')
image_parser.add_argument('image_path', type=str, help='Шлях до зображення')

args = parser.parse_args()

if args.command == 'image':
    result = predict_from_image(args.api_url, args.image_path)
    if result:
        print("Prediction result:")
        pprint.pprint(result)
else:
    parser.print_help()
"""
