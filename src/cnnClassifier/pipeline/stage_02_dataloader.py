from cnnClassifier import logger
from cnnClassifier.components.dataloader import Dataloader
from cnnClassifier.entity.config_entity import DataLoaderConfig
from cnnClassifier.config.configuration import ConfigurationManager


class DataLoaderPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        dataloader_config = config.get_dataloader_config()
        dataloader = Dataloader(config=dataloader_config)
        train_dataloader, test_dataloader, classes = dataloader.create_dataloader()
        return train_dataloader, test_dataloader, classes
    

if __name__ == '__main__':
    STAGE_NAME = "dataloader"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        dataloader = DataLoaderPipeline()
        dataloader.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        raise