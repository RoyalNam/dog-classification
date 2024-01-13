from cnnClassifier.pipeline.state_02_dataloader import DataLoaderPipeline
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_model import PrepareModel
from cnnClassifier import logger


class PrepareModelPipeline:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
    
    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.get_prepare_model_config()
        prepare_model = PrepareModel(config=prepare_model_config, num_classes=self.num_classes)
        prepare_model.get_base_model()
        prepare_model.updated_model()


if __name__ == '__main__':
    STAGE_NAME = "prepare model"
    try:
        dataloader = DataLoaderPipeline()
        _, _, classes = dataloader.main()
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        prepare_model = PrepareModelPipeline(num_classes=len(classes))
        prepare_model.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        raise