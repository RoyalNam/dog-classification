import torch
from torch import nn
import torchvision
from torchsummary import summary
from cnnClassifier.utils.common import save_model
from cnnClassifier.entity.config_entity import PrepareModelConfig


class PrepareModel:
    def __init__(self, config: PrepareModelConfig, num_classes: int=2) -> None:
        self.config = config
        self.model = None
        self.num_classes = num_classes
        
    def get_base_model(self):
        weights = torchvision.models.get_weight(f"{self.config.params_weights}.DEFAULT")
        self.model = torchvision.models.get_model(self.config.params_model, weights=weights)
        
        save_model(self.model, self.config.base_model_path)
    
    def _prepare_full_model(self, num_classes, seed = 42):
        torch.manual_seed(seed=seed)
        if self.model == None:
            self.get_base_model()
        self.model.classifier = torch.nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.LazyLinear(out_features=num_classes)
        )
        summary(self.model, input_size=(3, 224, 224))
        return self.model

    def updated_model(self):
        full_model = self._prepare_full_model(self.num_classes, self.num_classes)
        
        save_model(full_model, self.config.updated_model_path)