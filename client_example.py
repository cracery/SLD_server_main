import requests
import base64
import json
import argparse
from pathlib import Path
import pprint

def predict_from_image(api_url, image_path):
    """
    Відправляє запит на прогнозування рівня стресу за зображенням
    
    Args:
        api_url: URL API сервера
        image_path: Шлях до зображення
    
    Returns:
        Словник з результатами прогнозування
    """
    # Перевірка існування файлу
    if not Path(image_path).exists():
        print(f"Помилка: Файл {image_path} не існує")
        return None
    
    # Підготовка файлу для відправки
    files = {
        'file': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        # Відправка запиту
        response = requests.post(f"{api_url}/predict/image", files=files)
        
        # Закриття файлу
        files['file'][1].close()
        
        # Обробка відповіді
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Помилка: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Помилка при відправці запиту: {str(e)}")
        return None

def predict_from_base64(api_url, image_path):
    """
    Відправляє запит на прогнозування рівня стресу за base64-кодованим зображенням
    
    Args:
        api_url: URL API сервера
        image_path: Шлях до зображення
    
    Returns:
        Словник з результатами прогнозування
    """
    # Перевірка існування файлу
    if not Path(image_path).exists():
        print(f"Помилка: Файл {image_path} не існує")
        return None
    
    # Кодування зображення у base64
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        # Відправка запиту
        response = requests.post(
            f"{api_url}/predict/base64", 
            data={'base64_image': base64_image}
        )
        
        # Обробка відповіді
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Помилка: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Помилка при відправці запиту: {str(e)}")
        return None

def predict_from_emotions(api_url, emotions):
    """
    Відправляє запит на прогнозування рівня стресу за вектором емоцій
    
    Args:
        api_url: URL API сервера
        emotions: Список з 8 значень ймовірностей емоцій
    
    Returns:
        Словник з результатами прогнозування
    """
    if len(emotions) != 8:
        print("Помилка: Вектор емоцій повинен містити 8 значень")
        return None
    
    try:
        # Відправка запиту
        response = requests.post(
            f"{api_url}/predict/emotions", 
            json={'emotions': emotions}
        )
        
        # Обробка відповіді
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Помилка: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Помилка при відправці запиту: {str(e)}")
        return None

def get_model_info(api_url):
    """
    Отримує інформацію про модель
    
    Args:
        api_url: URL API сервера
    
    Returns:
        Словник з інформацією про модель
    """
    try:
        response = requests.get(f"{api_url}/model_info")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Помилка: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Помилка при відправці запиту: {str(e)}")
        return None

def display_prediction_with_metrics(result):
    """
    Відображає результат прогнозування з метриками.
    """
    print(f"Рівень стресу: {result['stress_level']}")
    print(f"Впевненість: {result['confidence']:.2f}%")
    print("Ймовірності:")
    for level, prob in result['probabilities'].items():
        print(f"  {level}: {prob:.2f}%")
    
    if 'metrics' in result and result['metrics']:
        print("\nМетрики для класу:")
        for metric_name, value in result['metrics'].items():
            print(f"  {metric_name}: {value:.2f}")
    
    print("\n")

def get_model_metrics(api_url):
    """
    Отримує детальні метрики моделі.
    """
    try:
        response = requests.get(f"{api_url}/model_metrics")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Помилка: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Помилка при відправці запиту: {str(e)}")
        return None
    
def main():
    parser = argparse.ArgumentParser(description='Клієнт для API виявлення стресу')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                        help='URL API сервера (за замовчуванням: http://localhost:8000)')
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Парсер для команди "image"
    image_parser = subparsers.add_parser('image', help='Прогнозування за зображенням')
    image_parser.add_argument('image_path', type=str, help='Шлях до зображення')
    
    # Парсер для команди "base64"
    base64_parser = subparsers.add_parser('base64', help='Прогнозування за base64-кодованим зображенням')
    base64_parser.add_argument('image_path', type=str, help='Шлях до зображення для кодування у base64')
    
    # Парсер для команди "emotions"
    emotions_parser = subparsers.add_parser('emotions', help='Прогнозування за вектором емоцій')
    emotions_parser.add_argument('emotions', type=float, nargs=8, 
                                help='8 значень ймовірностей емоцій (angry, disgust, fear, happy, sad, surprise, neutral, contempt)')
    
    # Парсер для команди "info"
    info_parser = subparsers.add_parser('info', help='Отримання інформації про модель')
    
    # Парсер для команди "metrics"
    metrics_parser = subparsers.add_parser('metrics', help='Отримання детальних метрик моделі')

    args = parser.parse_args()
    
    if args.command == 'image':
        result = predict_from_image(args.api_url, args.image_path)
        if result:
            print("Результат прогнозування:")
            pprint.pprint(result)
    
    elif args.command == 'base64':
        result = predict_from_base64(args.api_url, args.image_path)
        if result:
            print("Результат прогнозування:")
            pprint.pprint(result)
    
    elif args.command == 'emotions':
        result = predict_from_emotions(args.api_url, args.emotions)
        if result:
            print("Результат прогнозування:")
            pprint.pprint(result)
    
    elif args.command == 'info':
        result = get_model_info(args.api_url)
        if result:
            print("Інформація про модель:")
            pprint.pprint(result)
    
    elif args.command == 'metrics':
        result = get_model_metrics(args.api_url)
        if result:
            print("Метрики моделі:")
            for metric_name, values in result['metrics'].items():
                print(f"\n{metric_name.upper()}:")
                if isinstance(values, dict):
                    for class_name, value in values.items():
                        print(f"  {class_name}: {value:.2f}")
                else:
                    print(f"  {values:.2f}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()