# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of this project is to predict which credit card customers are most likely to "churn" and to find out which features have the biggest impact on that.

## Files and data description
To start working on the current project it is needed to set up following directory structure:
```
├── data
|   ├── bank_data.csv
├── images
|   ├── eda
|   ├── results
├── logs
├── models
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── constants.py
├── Guide.ipynb
├── README.md
└── requirements_py3.8.txt
```

Dataset for this project was pulled from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). Dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

## Running Files
To reproduce the results of the project follow the steps described below:

 1. Run `pip install requirements.txt` to install all libraries that are necessary to work on the project.
 2. Run the main file that processes the data from `data/bank_data.csv`  and trains the models. Use command: `python churn_library.py`.
If you want to change default values of pathes where to store output results, please check `constants.py`. After script finishes you should find images with eda plots in `images/eda` and model summary in `images/results`.
3.Run the logging and testing script with command: `python churn_script_logging_and_tests.py`. After it finishes you should find the results of testing process in `logs/churn_library.log` file.


