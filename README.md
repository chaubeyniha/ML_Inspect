# ML_Inspect

## Description

An example pipeline has been built with a dataset found on Kaggle.
https://www.kaggle.com/xiaowenlimarketing/international-student-time-management

## Requirements

Python 3.8+ with pandas and scikit-learn.

## Usage

To run the pipeline:

```shell
python3.8 -m venv venv
source venv/bin/activate

pip install -U pip
pip install pandas scikit-learn

python Psych_Pipeline/pipeline_1.py
```

To explore the data:

```shell
pip install jupyter seaborn matplotlib

jupyter notebook
```

## Pipeline Screening

We screen the pipelines for potential issues using [`ArgusEyes`](https://github.com/schelterlabs/arguseyes). `ArgusEyes` requires Python 3.9 (Python 3.8 will raise an error during installation), so we start with a new virtual env (Note: This is not necessary if you already used Python 3.9 for the previous part).

```shell
python3.9 -m venv argusenv
source argusenv/bin/activate

pip install -U pip git+https://github.com/schelterlabs/arguseyes.git@windows

cd Example_Pipeline/

# Screen decision tree pipeline
python -m arguseyes time-management-decision-tree.yaml
# View fairness metrics
mlflow ui --backend-store-uri ../mlruns

# Screen logistic regression pipeline
python -m arguseyes time-management-logistic.yaml
# View fairness metrics
mlflow ui --backend-store-uri ../mlruns
```

With the `mlflow` server running, open <http://localhost:5000> in your browser to view the calculated fairness metrics.
