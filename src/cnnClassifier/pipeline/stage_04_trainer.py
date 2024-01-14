from cnnClassifier.components.trainer import Trainer
from cnnClassifier.pipeline.stage_02_dataloader import DataLoaderPipeline
from cnnClassifier.config.configuration import ConfigurationManager
import torch
from cnnClassifier import logger

class TrainerPipeline:
    def __init__(self, train_dataloader, test_dataloader) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
    
    def main(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = ConfigurationManager()
        trainer_config = config.get_trainer_config()
        trainer = Trainer(config=trainer_config, train_dataloader=self.train_dataloader, test_dataloader=self.test_dataloader, device=device)
        result = trainer.train()
        trainer.log_into_mlflow()
        print(result)
        
        
if __name__ == '__main__':
    STAGE_NAME = "trainer"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
        dataloader = DataLoaderPipeline()
        train_dataloader, test_dataloader, _ = dataloader.main()
        
        trainer = TrainerPipeline(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        trainer.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=========x")
    except Exception:
        raise
