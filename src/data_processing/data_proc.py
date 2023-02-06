import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
from joblib import dump

# logging.basicConfig(
#     filename='./logs/results.log',
#     level=logging.INFO,
#     filemode='a+',
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt="%Y-%m-%d %H:%M:%S")


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
    col_input_clean = ["workclass", "education",
                       "marital_status", "occupation",
                       "relationship", "race",
                       "sex", "native_country"]
    try:
        print('err')
        df = pd.read_csv(f"{root_path}/data/clean/census.csv")

    except Exception as exp_error:
        print(exp_error)
        df = pd.read_csv(f"{root_path}/data/raw/census.csv")
        df.columns = [c.strip().replace('-', '_') for c in df.columns]
        df.replace({'?': None}, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(["education_num", "capital_gain",
                "capital_loss"], axis=1, inplace=True)

        # clean space in names
        for i in col_input_clean:
            df[i] = df[i].apply(lambda x: x.strip())
        df.to_csv(f"{root_path}/data/clean/census.csv", index=False)

        df.rename(columns={'hoursPerWeek': 'hours-per-week',
                           "nativeCountry": "native-country"})
        print('here')
    df__train, df__test = train_test_split(df,
                                           test_size=0.20,
                                           random_state=10,
                                           stratify=df['salary']
                                           )

    return df__train, df__test


def process_data(X,
                 root_path=None,
                 categorical_features=[],
                 label=None,
                 training=True,
                 encoder=None,
                 lb=None,
                 ):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    root_path: str:
        String path to the root folder
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    if training is True:
        dump(encoder, f"{root_path}/model_output/encoder.joblib")
        dump(lb, f"{root_path}/model_output/lb.joblib")
        dff = pd.DataFrame(X)
        dff.to_csv(f"{root_path}/model_output/tt.csv")

    return X, y, encoder, lb
