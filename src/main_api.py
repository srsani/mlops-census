"""
Main script to instantiate an FastAPI instance
author: srsani
Date: 2023/01/27
"""
import os
import yaml
import joblib

from fastapi import FastAPI
import pandas as pd

from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.data_processing.data_proc import process_data
# from src.model_validation.model_valid import run_inference

import sys
sys.path.append('..')


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class ModelInput(BaseModel):
    age: int = Field(..., example=45)
    education: str = Field(..., example="Bachelors")
    fnlgt: int = Field(..., example=2334)
    hoursPerWeek: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    nativeCountry: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True

# class ModelInput(BaseModel):
#     age: int
#     workclass: Literal['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
#                        'Local-gov', 'Self-emp-inc', 'Without-pay']
#     fnlgt: int
#     education: Literal['Bachelors', 'HS-grad', '11th', 'Masters',
#                        '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc',
#                        '7th-8th', 'Doctorate', 'Prof-school',
#                        '5th-6th', '10th', '1st-4th', 'Preschool',
#                        '12th']
#     marital_status: Literal['Never-married', 'Married-civ-spouse', 'Divorced',
#                             'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
#                             'Widowed']
#     occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
#                         'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
#                         'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
#                         'Craft-repair', 'Protective-serv', 'Armed-Forces',
#                         'Priv-house-serv']
#     relationship: Literal['Not-in-family', 'Husband', 'Wife', 'Own-child',
#                           'Unmarried', 'Other-relative']
#     race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
#                   'Other']
#     sex: Literal['Male', 'Female']
#     hoursPerWeek: int
#     nativeCountry: Literal['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
#                            'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
#                            'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
#                            'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
#                            'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
#                            'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
#                            'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
#                            'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
#                            'Holand-Netherlands']


def run_inference(data, cat_features):
    """Load model and run inference
    Parameters
    ----------
    root_path
    data
    cat_features

    Returns
    -------
    prediction
    """
    model = joblib.load("src/model_output/model.joblib")
    encoder = joblib.load("src/model_output/encoder.joblib")
    lb = joblib.load("src/model_output/lb.joblib")

    X, _, _, _ = process_data(X=data,
                              categorical_features=cat_features,
                              encoder=encoder,
                              lb=lb,
                              training=False)
    preds = model.predict(X)
    prediction = lb.inverse_transform(preds)[0]

    return prediction.strip()


with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = joblib.load("src/model_output/model.joblib")
    encoder = joblib.load("src/model_output/encoder.joblib")
    lb = joblib.load("src/model_output/lb.joblib")
    print(model)


@ app.get("/")
async def get_items():
    # print(model)
    return {"message": "Greetings"}


@ app.post("/")
async def inference(input_data: ModelInput):
    """
    invocates inference pipeline for single prediction

    Args:
        input_data (ModelInput): data dictionary
    """
    change_keys = config['inference']['update_keys']
    columns = config['inference']['columns']
    cat_features = config['data']['cat_features']

    input_data = input_data.dict()
    for new_key, old_key in change_keys:
        try:
            input_data[new_key] = input_data.pop(old_key)
        finally:
            pass

    df = pd.DataFrame.from_dict([input_data], orient='columns')
    df = df[columns]
    prediction = run_inference(df, cat_features).strip()

    return {"prediction":
            prediction}
