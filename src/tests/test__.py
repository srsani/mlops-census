'''
3 Tests code for the the api
4 tests for model training
author: srsani
Date: 2023/02/05
'''

import os
import pathlib
from src.data_processing.data_proc import get_train_test
from src.model_training.train_model import process_train_save_model
from src.model_validation.model_valid import val_model
from src.main_api import run_inference

import sys
sys.path.append('..')


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings"}


def test_post_low(client):
    request = client.post("/", json={'age': 33,
                                     'workclass': 'Private',
                                     'fnlgt': 149184,
                                     'education': 'HS-grad',
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 60,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {'prediction': '<=50K'}


def test_post_high(client):
    request = client.post("/", json={'age': 42,
                                     'workclass': 'Private',
                                     'fnlgt': 159449,
                                     'education': 'Bachelors',
                                     'marital_status': 'Married-civ-spouse',
                                     'occupation': 'Exec-managerial',
                                     'relationship': 'Husband',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 40,
                                     'nativeCountry': 'United-States'})
    assert request.status_code == 200
    assert request.json() == {'prediction': '>50K'}


def test_post_malformed(client):
    r = client.post("/", json={
        "age": 12,
        "workclass": "",
        "education": "Some-college",
        "maritalStatus": "",
        "occupation": "",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 422


def test_get_train_test_data():
    train_df, test_df = get_train_test(f'{pathlib.Path().resolve()}/src/')

    assert train_df.shape[0] == 26029
    assert train_df.shape[1] == 12

    assert test_df.shape[0] == 6508
    assert test_df.shape[1] == 12


def test_train_save_model(clean_data, cat_features):

    org_path = f'{pathlib.Path().resolve()}/src/'
    process_train_save_model(train=clean_data,
                             cat_features=cat_features,
                             root_path=org_path)

    assert os.path.isfile(f'{org_path}model_output/model.joblib')
    assert os.path.isfile(f'{org_path}model_output/encoder.joblib')
    assert os.path.isfile(f'{org_path}model_output//lb.joblib')


def test_run_inference_low(inference_data_low, cat_features):
    prediction = run_inference(inference_data_low,
                               cat_features=cat_features)

    assert prediction == "<=50K"


def test_run_inference_high(inference_data_high, cat_features):
    prediction = run_inference(inference_data_high,
                               cat_features=cat_features)

    assert prediction == ">50K"
