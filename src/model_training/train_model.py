"""
Script to train model.
author: srsani
Date: 2023/01/23
"""
from src.data_processing.data_proc import process_data
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import logging
from model_training.model_code import train_model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


logging.basicConfig(
    filename='./logs/results1.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def get_train_test(root_path):
    """ Import the cleaned data and split it to train and tes

    Inputs
    ------
    root_path:  String
                Path to the data folder

    Returns
    ------
    df__train: pd.DataFrame
                DataFrame for future model training

    df__test: pd.DataFrame
                DataFrame for future test
    """
    df = pd.read_csv(f"{root_path}/data/clean/census.csv")
    df__train, df__test = train_test_split(df,
                                           test_size=0.20,
                                           random_state=10,
                                           stratify=df['salary']
                                           )

    return df__train, df__test


def process_train_save_model(train, cat_features, root_path):
    """ Train the full model

    Inputs
    ------
    train
    cat_features
    root_path

    Returns
    ------

    """
    X_train, y_train, encoder, lb = process_data(X=train,
                                                 root_path=root_path,
                                                 categorical_features=cat_features,
                                                 label="salary",
                                                 training=True
                                                 )
    # train model
    trained_model = train_model(X_train, y_train)
    # save model artifacts
    dump(trained_model, f"{root_path}/model_output/model.joblib")
    dump(encoder, f"{root_path}/model_output/encoder.joblib")
    dump(lb, f"{root_path}/model_output/lb.joblib")
