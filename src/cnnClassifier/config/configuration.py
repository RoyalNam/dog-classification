from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_dir
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataLoaderConfig,
    PrepareModelConfig,
    TrainerConfig
)


class ConfigurationManager:
    def __init__(self) -> None:
        config_path = CONFIG_PATH
        params_path = PARAMS_PATH

        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_dir([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']

        create_dir([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config['root_dir'],
            source_url=config['source_url'],
            local_data_file=config['local_data_file'],
            unzip_dir=config['unzip_dir']
        )
        return data_ingestion_config

    def get_dataloader_config(self) -> DataLoaderConfig:
        config = self.config['dataloader']
        create_dir([config['root_dir']])

        dataloader_config = DataLoaderConfig(
            root_dir=config['root_dir'],
            train_dir=config['train_dir'],
            test_dir=config['test_dir'],
            image_transforms_path=config['image_transforms_path'],
            weights=self.params['WEIGHTS'],
            params_batch_size=self.params['BATCH_SIZE']
        )
        return dataloader_config
    
    def get_prepare_model_config(self):
        config = self.config['prepare_model']
        create_dir([config['root_dir']])
        
        prepare_model_config = PrepareModelConfig(
            root_dir=config['root_dir'],
            base_model_path=config['base_model_path'],
            updated_model_path=config['updated_model_path'],
            params_weights=self.params['WEIGHTS'],
            params_model=self.params['MODEL'],
            params_seed=self.params['SEED']
        )
        return prepare_model_config
    
    def get_trainer_config(self) -> TrainerConfig:
        config = self.config['trainer']
        params = self.params
        create_dir([config['root_dir']])
        
        trainer_config = TrainerConfig(
            root_dir=config['root_dir'],
            trained_model_path=config['trained_model_path'],
            updated_model_path=config['updated_model_path'],
            mlflow_uri=config['mlflow_uri'],
            params_epochs=params['EPOCHS'],
            params_lr=params['LEARNING_RATE'],
            all_params=params
        )
        return trainer_config
