import zipfile
import os
import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_data(self):
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='

            gdown.download(prefix + file_id, str(zip_download_dir))
            logger.info(f"Download data from {dataset_url} into file {zip_download_dir} successfully")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e

    def extract_zip_file(self):
        zip_download_dir = self.config.local_data_file
        unzip_dir = self.config.unzip_dir
        try:
            os.makedirs(unzip_dir, exist_ok=True)
            with zipfile.ZipFile(zip_download_dir, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            logger.info(f"Successfully extracted {zip_download_dir} to {unzip_dir}")
        except Exception as e:
            logger.error(f"Error: {e} during zip file extraction")
            raise e
