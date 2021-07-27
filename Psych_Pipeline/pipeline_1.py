import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read the data
raw_data = pd.read_csv("International students Time management data.csv")
pd.set_option("display.max.columns", None)

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

# transform y variable (T/F)
raw_data['label'] = (raw_data['7'] == 'Agree') | (raw_data['7'] == 'Strong Agree')

# create pipeline model
pipe = Pipeline([
    ('features', categorical_prepr),
    ('classifier', LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(raw_data, raw_data['label'], random_state=1)

# fit the pipeline to the training data
pipe.fit(X_train, y_train)

# predict target values on training data
pred_train = pipe.predict(X_train)

# validate with X
pred_test = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
print("Train score: ", pipe.score(X_train, y_train))
print("Test score: ", score)


