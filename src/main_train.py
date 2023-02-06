"""
Main script to train an RF model FastAPI instance
author: srsani
Date: 2023/01/27
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import logging
from omegaconf import DictConfig
import hydra
from src.data_processing.data_proc import get_train_test
from src.model_training.train_model import process_train_save_model
from src.model_validation.model_valid import val_model


logging.basicConfig(
    filename='./logs/results2.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


_steps = [
    "data_processing",
    "train_model",
    "check_score"
]


@hydra.main(config_name="config.yaml")
def go(config: DictConfig):
    """
    Run pipeline stages
    """

    root_path = hydra.utils.get_original_cwd()
    logging.info(root_path)
    # Steps to execute
    cat_features = config['data']['cat_features']

    if "data_processing" in _steps:
        logging.info("data_processing step")
        train_df, test_df = get_train_test(root_path)
        logging.info(f"train_shape: {train_df.shape}")
        logging.info(f"test_df: {test_df.shape}")

    if "train_model" in _steps:
        logging.info("Train model: ")
        process_train_save_model(train_df, cat_features, root_path)

    if "check_score" in _steps:
        logging.info(" check_score step")
        val_model(test_df=test_df,
                  cat_features=cat_features,
                  root_dir=root_path)


if __name__ == "__main__":

    go()
