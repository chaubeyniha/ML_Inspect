import pandas as pd
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

ord = OrdinalEncoder()
raw_data[['7']] = ord.fit_transform(raw_data[['7']])

le = LabelEncoder()
raw_data['Course'] = le.fit_transform(raw_data['Course'])
raw_data['Academic'] = le.fit_transform(raw_data['Academic'])
raw_data['Attendance'] = le.fit_transform(raw_data['Attendance'])

# transform multiple columns
features_col = ['Course', 'Academic', 'Attendance']
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_prepr = ColumnTransformer([
    ('imputer', SimpleImputer(strategy='most_frequent'), features_col),
    ('one-hot-encoder', enc, features_col)])

# create pipeline model
pipe = Pipeline([
    ('features', categorical_prepr),
    ('classifier', LogisticRegression())
])

# initialize variable X and y (column 7 because no missing values)
X = raw_data[features_col]
y = raw_data[['7']]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=1)

# fit the pipeline to the training data
pipe.fit(X_train, Y_train.values.ravel())

# predict target values on training data
pipe.predict(X_train)

# validate with X
validate_predication = pipe.predict(X_test)
difference = mean_absolute_error(Y_test, validate_predication)
score = pipe.score(X_test, Y_test)
print(difference)
print(score)