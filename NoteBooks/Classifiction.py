import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


path = kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")

df_ai4i = pd.read_csv(f'{path}/ai4i2020.csv')
rows, cols = df_ai4i.shape
df_ai4i=df_ai4i.drop(columns='UDI')
df_ai4i.columns

missing = df_ai4i.isnull().sum()
# print('Missing values per row:\n',missing,'\n')
df_ai4i[df_ai4i.isnull().any(axis=1)]



# print(df_ai4i.duplicated().sum())
df_ai4i[df_ai4i.duplicated()]


num_cols= df_ai4i.select_dtypes(include=np.number).columns
cat_cols= df_ai4i.select_dtypes(include='object').columns


# for col in num_cols:
#     plt.figure(figsize=(5, 2))
#     plt.boxplot(df_ai4i[col], vert=False)
#     plt.title(col)
#     plt.show()


# for col in num_cols:
#     Q1 = df_ai4i[col].quantile(0.25)
#     Q3 = df_ai4i[col].quantile(0.75)
#     IQR = Q3 - Q1

#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR

#     outliers = df_ai4i[(df_ai4i[col] < lower) | (df_ai4i[col] > upper)]

#     print(f"{col}: {len(outliers)} outliers")


# for col in df_ai4i.columns:
#     print(f"{col}: {df_ai4i[col].nunique()}\n")


df_ai4i=df_ai4i.drop(columns='Product ID')
df_ai4i.columns

df_ai4i['Type'].unique()
# M = Medium, L = Low, H = High --> quality of the equipment
# can deal with it using numbers that represents the levels

level_map = {
    "L": 0,
    "M": 1,
    "H": 2
}

df_ai4i["Type"] = df_ai4i["Type"].map(level_map)
# df_ai4i.info()

failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

def get_failure_type(row):
    for col in failure_cols:
        if row[col] == 1:
            return col
    return 'No Failure'

# finall idea here is to remove No Failure from the dataset and just classifiy the Failure_Type (becouse Wiam Finshed with good accuracy for Binary Class)
df_ai4i["Failure_Type"] = df_ai4i.apply(get_failure_type, axis=1)
df_failure = df_ai4i[df_ai4i["Failure_Type"] != "No Failure"].copy()
X = df_failure.drop(columns=[ "Machine failure", "Failure_Type", "TWF", "HDF", "PWF", "OSF", "RNF"])
y = df_failure["Failure_Type"]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# i try smote with RandomForest but not working will 
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

model = BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))



