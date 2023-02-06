import pandas as pd
import pytest
from fastapi.testclient import TestClient
import yaml
import pathlib

from src.main_api import app

import sys
sys.path.append('../..')


@pytest.fixture
def client():
    """
    Get test client
    """
    api_client = TestClient(app)
    return api_client


@pytest.fixture
def clean_data():
    """
    Get clean
    """
    df = pd.read_csv(f'{pathlib.Path().resolve()}/src/data/clean/census.csv')
    return df


@pytest.fixture
def cat_features():
    """
    Get dataset
    """
    with open('src/config.yaml') as f:
        config = yaml.safe_load(f)

    return config['data']['cat_features']


@pytest.fixture
def inference_data_low():
    data_dict = {'age': 19,
                 'workclass': 'Private',
                 'fnlgt': 77516,
                 'education': 'HS-grad',
                 'marital_status': 'Never-married',
                 'occupation': 'Own-child',
                 'relationship': 'Husband',
                 'race': 'Black',
                 'sex': 'Male',
                 'hours-per-week': 40,
                 'native_country': 'United-States'
                 }
    return pd.DataFrame.from_dict([data_dict], orient='columns')


@pytest.fixture
def inference_data_high():
    data_dict = {'age': 42,
                 'workclass': 'Private',
                 'fnlgt': 159449,
                 'education': 'Bachelors',
                 'marital_status': 'Married-civ-spouse',
                 'occupation': 'Exec-managerial',
                 'relationship': 'Husband',
                 'race': 'White',
                 'sex': 'Male',
                 'hours_per_week': 40,
                 'native_country': 'United-States'}
    return pd.DataFrame.from_dict([data_dict], orient='columns')
