# Model Card

For additional information, see the [Model Card](https://arxiv.org/pdf/1810.03993.pdf) paper.

## Model Details
I've used five-fold cross-validation in this work to select the best mode.    A RandomForestClassifier with the best accuracy score of (0.85) was trained.

```bash
RandomForestClassifier(criterion='entropy', min_samples_split=40, n_jobs=-1)
```

## Intended Use



## Training Data

The raw dataset is sourced from [here](https://archive.ics.uci.edu/ml/datasets/census+income) wit the following columns:
:

```
 columns:
    [ 'age',
      'workclass',
      'fnlgt',
      'education',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'sex',
      'hours-per-week',
      'native-country'
    ]
```

```
cat_features : [
      "workclass",
      "education",
      "marital_status",
      "occupation",
      "relationship",
      "race",
      "sex",
      "native_country",
  ]
```
## Evaluation Data

A 20% slice of the data source is used to validate the model. Also a slice report is located under "src/model_output/slice_output.txt" 

## Metrics

The model was evaluated using Accuracy score, F1 beta score, Precision and Recall. The average cross-validation accuracy score is 86%

## Ethical Considerations

The Ethical consideration was checked using different data slices. 
further investigation is needed

## Caveats and Recommendations

- Improve the gender biases
- Investigate the imbalance data in many of - the categorical features