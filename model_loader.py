import os
import torch
import numpy as np
import json
from torch import nn

# Визначення класу FeatureWeightedNN
class FeatureWeightedNN(nn.Module):
    def __init__(self, input_dim=8, num_classes=3, feature_importance=None):
        super(FeatureWeightedNN, self).__init__()

        # Ініціалізація параметрів
        self.input_dim = input_dim

        # Шар зважування ознак
        if feature_importance is not None:
            # Нормалізація важливості ознак
            if isinstance(feature_importance, np.ndarray):
                normalized_importance = feature_importance / np.max(feature_importance) if np.max(feature_importance) > 0 else feature_importance
                self.feature_weights = nn.Parameter(
                    torch.tensor(normalized_importance, dtype=torch.float32).view(1, -1),
                    requires_grad=True
                )
            else:
                self.feature_weights = nn.Parameter(torch.ones(1, input_dim), requires_grad=True)
        else:
            self.feature_weights = nn.Parameter(torch.ones(1, input_dim), requires_grad=True)

        # Основна архітектура мережі
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Зважування вхідних ознак за їх важливістю
        weighted_x = x * self.feature_weights
        # Обробка зважених ознак моделлю
        return self.model(weighted_x)
    
def load_weighted_model(model_name):
    try:
        # Шляхи до файлів моделі
        model_path = f"models/{model_name}.pth"
        feature_importance_path = f"models/{model_name}_feature_importance.npy"
        weights_path = f"models/{model_name}_learned_weights.npy"
        metadata_path = f"models/{model_name}_metadata.json"
        
        print(f"Спроба завантажити модель з {model_path}")
        
        # Перевірка наявності файлів
        if not os.path.exists(model_path):
            print(f"Файл моделі не знайдено за шляхом {model_path}")
            return create_default_model()
        
        # Завантажуємо feature_importance
        feature_importance = None
        if os.path.exists(feature_importance_path):
            try:
                feature_importance = np.load(feature_importance_path, allow_pickle=True)
                print(f"Завантажено feature_importance, тип: {type(feature_importance)}")
            except Exception as e:
                print(f"Помилка при завантаженні feature_importance: {str(e)}")
        else:
            print(f"Файл feature_importance не знайдено")
        
        # Створюємо екземпляр моделі
        model = FeatureWeightedNN(input_dim=8, num_classes=3, feature_importance=feature_importance)
        
        # Завантажуємо state_dict моделі
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Відображаємо ключі для діагностики
        print(f"Ключі у state_dict: {list(state_dict.keys())}")
        
        # Спробуємо завантажити state_dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Модель успішно завантажена з strict=True")
        except Exception as e:
            print(f"Помилка при завантаженні з strict=True: {str(e)}")
            
            # Спробуємо з strict=False
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Модель завантажена з параметром strict=False")
            except Exception as e2:
                print(f"Помилка при завантаженні з strict=False: {str(e2)}")
                print("Неможливо завантажити модель, використовуємо дефолтну")
                return create_default_model()
        
        # Завантажуємо learned_weights
        learned_weights = None
        if os.path.exists(weights_path):
            try:
                learned_weights = np.load(weights_path, allow_pickle=True)
                print(f"Завантажено learned_weights, тип: {type(learned_weights)}")
            except Exception as e:
                print(f"Помилка при завантаженні learned_weights: {str(e)}")
        else:
            print(f"Файл learned_weights не знайдено")
        
        # Завантажуємо метадані
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"Завантажено метадані: {list(metadata.keys()) if metadata else 'порожні метадані'}")
            except Exception as e:
                print(f"Помилка при завантаженні метаданих: {str(e)}")
        else:
            print(f"Файл метаданих не знайдено за шляхом {metadata_path}")
        
        # Переводимо модель в режим оцінки
        model.eval()
        print("Модель успішно завантажена і переведена в режим оцінки")
        return model, feature_importance, learned_weights, metadata
        
    except Exception as e:
        print(f"Загальна помилка при завантаженні моделі {model_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None   