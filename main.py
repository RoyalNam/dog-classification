from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_dataloader import DataLoaderPipeline
from cnnClassifier.pipeline.stage_03_prepare_model import PrepareModelPipeline
from cnnClassifier.pipeline.stage_04_trainer import TrainerPipeline

STAGE_NAME = "data ingestion"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    raise e

STAGE_NAME = "dataloader"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    dataloader = DataLoaderPipeline()
    train_dataloader, test_dataloader, classes = dataloader.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception:
    raise

STAGE_NAME = "prepare model"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started")
    prepare_model = PrepareModelPipeline(num_classes=len(classes))
    prepare_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed")
except Exception:
    raise

STAGE_NAME = "trainer"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    trainer = TrainerPipeline(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    trainer.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
except Exception:
    raise