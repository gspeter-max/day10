!pip install kaggle  # Install Kaggle if not installed
from google.colab import files

# Upload your Kaggle API key (kaggle.json)
files.upload()

# Move the API key to the correct location
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Set Google Drive path for storage
drive_path = "/content/drive/MyDrive/Kaggle_Datasets/Titanic"

# Create directory in Google Drive
!mkdir -p "{drive_path}"

# Download the Titanic dataset directly to Google Drive
!kaggle competitions download -c titanic -p "{drive_path}"

# Unzip the dataset in the same location
!unzip "{drive_path}/titanic.zip" -d "{drive_path}"

import pandas as pd

train_path = "/content/drive/MyDrive/Kaggle_Datasets/Titanic/train.csv"

df = pd.read_csv(train_path)
df.head()

df.isnull().sum()

df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)
most_frequent = df['Cabin'].mode()
df['Cabin'] = df['Cabin'].fillna(most_frequent[0:2])
most_frequent_Embarked = df['Embarked'].mode()
df['Embarked'] = df['Embarked'].fillna(most_frequent_Embarked[0])
df.isnull().sum()


from sklearn.impute import SimpleImputer

numeric_imputers = SimpleImputer(strategy = 'mean')
categorical_imputers = SimpleImputer(strategy = 'most_frequent')

df['Age'] = numeric_imputers.fit_transform(df[['Age']])
df[['Cabin','Embarked']] = categorical_imputers.fit_transform(df[['Cabin','Embarked']])
df['Age'] = df['Age'].astype(int)


df['Fare'] = df['Fare'].round(2)
print(df['Fare'])


from sklearn.preprocessing import OneHotEncoder  , OrdinalEncoder

one_hot = OneHotEncoder(sparse_output= True)
one_hot_data = one_hot.fit_transform(df[['Embarked', 'Sex']])

target_encoder = OrdinalEncoder()
df[['Embarked','Sex','Cabin','Ticket']] = target_encoder.fit_transform(df[['Embarked','Sex','Cabin','Ticket']])
df[['Embarked', 'Sex']] = df[['Embarked', 'Sex']].astype(int)


from sklearn.preprocessing import StandardScaler , MinMaxScaler

stand = StandardScaler()
min_max = MinMaxScaler()

numerical_features = ['PassengerId', 'Survived', 'Pclass', 'SibSp',
       'Parch', 'Fare']
standard_data = stand.fit_transform(df[numerical_features])
df[numerical_features] = min_max.fit_transform(df[numerical_features])
df[numerical_features] = df[numerical_features].astype(int)


import numpy as np

correlation = df[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch',  'Fare', 'Embarked']].corr()
# tringular = correlation.where(np.triu(np.ones(correlation.shape),k = 1))
triu = correlation.where( np.triu(np.ones(correlation.shape), k = 1).astype(bool))
threshold = 0.50
columns_high = [column for column in triu.columns if any(triu[column] > threshold)] 
df = df.drop(columns = columns_high)

from statsmodels.stats.outliers_influence import variance_inflation_factor


vif  = pd.DataFrame()
vif['columns'] = df.columns
vif['vif'] = [variance_inflation_factor(df.values, i) for i  in range(df.shape[1])]
print(vif['vif0']) 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix 

x = df[['PassengerId', 'Pclass', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier( n_estimators= 100, max_depth = 2, n_jobs = 2)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

classificaion_report = classification_report(y_test, y_predict)
print(f"classification report : {classificaion_report}")

confusion_matrix = confusion_matrix(y_test, y_predict)
print(f"confusion matrix  : {confusion_matrix}")

from sklearn.model_selection import GridSearchCV

grid_param = {
    'n_estimators' : [100, 150, 200,300,400,500], 
    'max_depth' : [3,6,7,6,8,9]
}

rf_model = RandomForestClassifier(  random_state = 42 , n_jobs = 2 )
grid_model = GridSearchCV(
    estimator = rf_model , 
    param_grid = grid_param, 
    cv = 3, 
    verbose = 2, 
    n_jobs = 2, 
    scoring = 'precision'
)

grid_model.fit(x_train, y_train)


best_estimators = grid_model.best_estimator_ 
best_prediction = best_estimators.predict(x_test)
print(best_estimators)
print(f"classification report best  : {classification_report(y_test, best_prediction)}")
