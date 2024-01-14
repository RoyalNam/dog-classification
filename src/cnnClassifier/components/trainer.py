from cnnClassifier.utils.common import save_model, save_json
import torch
from tqdm.auto import tqdm
from cnnClassifier.entity.config_entity import TrainerConfig
import mlflow
from urllib.parse import urlparse
from pathlib import Path
import os


class Trainer:
    def __init__(self, config: TrainerConfig, train_dataloader, test_dataloader, device='cpu') -> None:
        self.config = config
        self.device = device
        self.model = torch.load(self.config.updated_model_path, map_location=device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.params_lr)

        os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/RoyalNam/dog-classification.mlflow'
        os.environ['MLFLOW_TRACKING_USERNAME'] = "RoyalNam"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "118907a7c7d97e75baf459e59483bc4b508be688"

    def _train_epoch(self, model: torch.nn.Module, train_dataloader, criterion, optimizer, device):
        accs, losses = [], []
        model.train()
        for _, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            total_acc = (pred.argmax(1)==y).sum().item()
            acc = total_acc / y.size(0)
            accs.append(acc)
        
        train_acc = sum(accs) / len(accs)
        train_loss = sum(losses) / len(losses)
        return train_acc, train_loss
    
    def _test_epoch(self, model: torch.nn.Module, test_dataloader, criterion, device):
        accs, losses = [], []
        model.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                losses.append(loss.item())
                
                total_acc = (pred.argmax(1) == y).sum().item()
                acc = total_acc / y.size(0)
                accs.append(acc)
        test_acc = sum(accs) / len(accs)
        test_loss = sum(losses) / len(losses)
        return test_acc, test_loss

    def train(self):
        train_accs, train_losses = [], []
        test_accs, test_losses = [], []
        for epoch in tqdm(range(self.config.params_epochs)):
            train_acc, train_loss = self._train_epoch(self.model, self.train_dataloader, self.criterion, self.optimizer, self.device)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            test_acc, test_loss = self._test_epoch(self.model, self.test_dataloader, self.criterion, self.device)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            print(f"-> epoch {epoch+1} | train_acc: {train_acc} | train_loss: {train_loss} | test_acc: {test_acc} | test_loss: {test_loss}")
            
        self.result = {
            "train_acc": train_accs,
            "train_loss": train_losses,
            "test_acc": test_accs,
            "test_loss": test_losses,
        }
        self.scores = {"loss": self.result['train_acc'][-1], "accuracy": self.result['train_loss'][-1]}

        save_model(self.model.state_dict(), self.config.trained_model_path)
        save_json(path=Path("scores.json"), data=self.scores)

        return self.result

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_registry_uri()).scheme
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)
            
            if tracking_url_type_store != 'file':
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="shufflenet_v2_x0_5")
            else:
                mlflow.pytorch.log_model(self.model, "model")