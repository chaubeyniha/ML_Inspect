import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

# read the data
raw_data = pd.read_csv(r"C:\Users\nchau\Documents\Internships\ML_Inspect\International students Time management data.csv")
pd.set_option("display.max.columns", None)
#print(raw_data['Academic'])
#print(raw_data['Attendance'])
#print(raw_data['Course'])

# prepare for one-hot-encoding
# le = LabelEncoder()
# raw_data['Course'] = le.fit_transform(raw_data['Course'])
# raw_data['Academic'] = le.fit_transform(raw_data['Academic'])
# raw_data['Attendance'] = le.fit_transform(raw_data['Attendance'])
# print(raw_data['Academic'])
# print(raw_data['Course'])

# print(raw_data.columns)

# transform multiple columns
features_col = ['Course','Academic','Attendance']
features_col_cat = ['Course']
features_col_ord = ['Academic', 'Attendance']

enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
ord = OrdinalEncoder()

prep_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

prep_pipeline2 = Pipeline([
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('ordinal', OrdinalEncoder())
])

categorical_prepr = ColumnTransformer([
    ("prep_pipeline", prep_pipeline, ['Course']),
    ("prep_pipeline2", prep_pipeline2, ['Academic', 'Attendance'])
    ])

# transform y variable
raw_data['Bin_Code'] = np.where(raw_data['7'].str.contains("Dis", "Neither"), 1, 0)

impu_1 = SimpleImputer(strategy='most_frequent').fit_transform(raw_data[['Academic']])
print(impu_1)
fit_1 = enc.fit_transform(impu_1)
print(fit_1)
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


X = raw_data[features_col]
y = raw_data['Bin_Code']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=1)

# fit the pipeline to the training data
pipe.fit(X_train, Y_train)
print(raw_data['Academic'])
#print(raw_data['Attendance'])
#print(raw_data['Course'])

# predict target values on training data
pipe.predict(X_train)

# validate with X
validate_predication = pipe.predict(X_test)
difference = mean_absolute_error(Y_test, validate_predication)
score = pipe.score(X_test, Y_test)
print(difference)
print(score)