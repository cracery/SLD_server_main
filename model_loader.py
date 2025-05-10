import os
import torch
import numpy as np
import json
from torch import nn

class FeatureWeightedNN(nn.Module):
    def __init__(self, input_dim=8, num_classes=3, ftr_importance=None):
        super(FeatureWeightedNN, self).__init__()
        self.input_dim = input_dim

        # Feature weighting layer
        if ftr_importance is not None:
            # Normalise importance of features
            if isinstance(ftr_importance, np.ndarray):
                norm_importance= ftr_importance / np.max(ftr_importance) if np.max(ftr_importance) > 0 else ftr_importance
                self.ftr_weights= nn.Parameter(
                    torch.tensor(norm_importance, dtype=torch.float32).view(1, -1),
                    requires_grad=True
                )
            else:
                self.ftr_weights= nn.Parameter(torch.ones(1, input_dim), requires_grad=True)
        else:
            self.ftr_weights= nn.Parameter(torch.ones(1, input_dim), requires_grad=True)

        # Main architecture
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
        weighted_x= x * self.ftr_weights
        return self.model(weighted_x)
    
def load_weighted_model(model_name):
    try:
        model_path= f"models/{model_name}.pth"
        ftr_importance_path= f"models/{model_name}_feature_importance.npy"
        wghts_path= f"models/{model_name}_learned_weights.npy"
        metadata_path= f"models/{model_name}_metadata.json"
        print(f"Trying to load model from {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at path {model_path}")
            return create_default_model()
        
        # load feature_importance
        ftr_importance= None
        if os.path.exists(ftr_importance_path):
            try:
                ftr_importance = np.load(ftr_importance_path, allow_pickle=True)
                print(f"Loaded feature_importance, type: {type(ftr_importance)}")
            except Exception as e:
                print(f"Error loading feature_importance: {str(e)}")
        else:
            print(f"feature_importance file not found")
        
        # Create model instance
        model= FeatureWeightedNN(input_dim=8, num_classes=3, ftr_importance=ftr_importance)
        
        # Load model state_dict
        state_dict= torch.load(model_path, map_location=torch.device('cpu'))
        
        # Display keys for diagnostics
        print(f"Keys in state_dict: {list(state_dict.keys())}")
        
        # Load state_dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model is loaded with strict=False")
        except Exception as e:
            print(f"Error loading with strict=True: {str(e)}")
            
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Model is loaded with strict=False")
            except Exception as e2:
                print(f"Error loading with strict=False: {str(e2)}")
                print("Cannot load the model, use the default one")
                return create_default_model()
        

        #Load learned_weights
        learned_wghts = None
        if os.path.exists(wghts_path):
            try:
                learned_wghts = np.load(wghts_path, allow_pickle=True)
                print(f"Loaded learned_weights, type: {type(learned_wghts)}")
            except Exception as e:
                print(f"Error loading learned_weights: {str(e)}")
        else:
            print(f"File learned_weights not found")
        

        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata= json.load(f)
                print(f"Metadata has been uploaded: {list(metadata.keys()) if metadata else 'порожні метадані'}")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
        else:
            print(f"Metadata file not found at path {metadata_path}")
        
        # put into eval mode
        model.eval()
        print("Model successfully loaded and put into evaluation mode")
        return model, ftr_importance, learned_wghts, metadata
        
    except Exception as e:
        print(f"General error when loading model {model_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None   
