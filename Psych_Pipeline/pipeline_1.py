import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read the data
raw_data = pd.read_csv("International students Time management data.csv")
pd.set_option("display.max.columns", None)

enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
ord = OrdinalEncoder()

impute_and_one_hot = Pipeline([
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

impute_and_ordinal = Pipeline([
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('ordinal', OrdinalEncoder())
])

categorical_prepr = ColumnTransformer([
    ("impute_and_one_hot", impute_and_one_hot, ['Course']),
    ("impute_and_ordinal", impute_and_ordinal, ['Academic', 'Attendance'])
])

# transform y variable
raw_data['label'] = (raw_data['7'] == 'Agree') | (raw_data['7'] == 'Strong Agree')

impu_1 = SimpleImputer(strategy='most_frequent').fit_transform(raw_data[['Academic']])
# print(impu_1)
fit_1 = enc.fit_transform(impu_1)
# print(fit_1)
#
# impu_2 = SimpleImputer(strategy='most_frequent').fit_transform(raw_data[['Attendance']])
# print(impu_2)
# fit_2 = enc.fit_transform(impu_2)
# print(fit_2)
#
# impu_3 = SimpleImputer(strategy='most_frequent').fit_transform(raw_data[['Course']])
# print(impu_3)
# fit_3 = ord.fit_transform(impu_3)
# print(fit_3)

# create pipeline model
pipe = Pipeline([
    ('features', categorical_prepr),
    ('classifier', LogisticRegression())
])

X_train, X_test, Y_train, Y_test = train_test_split(raw_data, raw_data['label'], random_state=1)

# fit the pipeline to the training data
pipe.fit(X_train, Y_train)

# predict target values on training data
pred_train = pipe.predict(X_train)

# validate with X
pred_test = pipe.predict(X_test)
score = pipe.score(X_test, Y_test)
print("Test score:", score)