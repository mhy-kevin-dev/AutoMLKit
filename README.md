# AutoMLKit

This app was written in python using `streamlit` and `pycaret` to perform machine learning tasks in an elegant manner.

## Requirements

- python 3.9+

- streamlit

- pycaret 
    - I encountered errors when installing the full version `pip install pycaret[full]`, so we may be missing the two methods: 'xgboost' and 'catboost'."

- check `requirements.txt`

## Quick start

```sh
streamlit run app.py
```

1. Import your training data (.csv) via WEB UI.
2. Choose whether to use regression or classification for training. (will do cross-validation automatically)
3. With a single click, This app will simultaneously test multiple machine learning methods and sort them according to their performances.
4. Finally, import your test data for prediction.

## Example Data

[Kaggle Playground Series Season 3, Episode 3: Tabular Classification with an Employee Attrition Dataset](https://www.kaggle.com/competitions/playground-series-s3e3/overview)

Files:

    - train.csv 

    - test.csv

Outputs:

    - predict.csv

## Screenshots
![](screenshots/upload.png)
![](screenshots/profilling.png)
![](screenshots/training.png)
![](screenshots/regression.png)
![](screenshots/predict.png)
