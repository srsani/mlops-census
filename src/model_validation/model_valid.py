from model_training.model_code import compute_model_metrics
from data_processing.data_proc import process_data
import joblib
from datetime import date
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('..')

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def compute_score_per_slice(trained_model, test, encoder,
                            lb, cat_features, root_path):
    """
    Compute score per category class slice
    Parameters
    ----------
    trained_model
    test
    encoder
    lb

    Returns
    -------

    """
    with open(f'{root_path}/model_output/slice_output.txt', 'w') as file:
        today = date.today()
        file.write(f"Date: {str(today)} \n")
        for category in cat_features:
            for cls in test[category].unique():
                temp_df = test[test[category] == cls]

                x_test, y_test, _, _ = process_data(X=temp_df,
                                                    root_path=root_path,
                                                    categorical_features=cat_features,
                                                    training=False,
                                                    label="salary",
                                                    encoder=encoder,
                                                    lb=lb)

                y_pred = trained_model.predict(x_test)

                precision, recall, fbeta = compute_model_metrics(
                    y_test, y_pred)

                metric_info = f"{category} - {cls} Precision: {precision} Recall: {recall} FBeta: {fbeta}"
                logging.info(metric_info)
                file.write(f"{metric_info} \n")
        file.write("="*100 + '\n')


def val_model(test_df, cat_features, root_dir):
    """ Performs model validation task

    Parameters
    ----------
    test_df : pd.DataFrame
        test DataFrame
    cat_features: list
        list of categorical data
    root_dir: str
        path to where model folder location

    Returns
    -------

    """

    trained_model = joblib.load(f"{root_dir}/model_output/model.joblib")
    encoder = joblib.load(f"{root_dir}/model_output/encoder.joblib")
    lb = joblib.load(f"{root_dir}/model_output/lb.joblib")

    compute_score_per_slice(trained_model,
                            test_df,
                            encoder,
                            lb,
                            cat_features,
                            root_dir)


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
                              encoder=encoder, lb=lb, training=False)
    preds = model.predict(X)
    prediction = lb.inverse_transform(preds)[0]

    return prediction
