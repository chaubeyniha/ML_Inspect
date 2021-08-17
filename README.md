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

To screen the pipeline for potential issues:

```shell
pip install git+https://github.com/schelterlabs/arguseyes.git
python -m arguseyes time-management-decision-tree.yaml
python -m arguseyes time-management-logistic.yaml
```

**NB**: `ArgusEyes` requires Python 3.9 (Python 3.8 will raise an error during installation).
