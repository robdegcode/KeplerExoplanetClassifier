import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("cumulative.csv") #reads from csv file


df.dtypes #this is how we picked important vs unimportant types

df = df[(df["koi_disposition"] != "CANDIDATE") & (df["koi_disposition"] != "NOT DISPOSITIONED")] #we only want to keep confirmed and false positives to train model on

df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int) # if confirmed -> true -> 1... if false pos -> false -> 0

df = df.drop(columns=["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_disposition", "koi_tce_delivname", "koi_teq_err1", "koi_teq_err2"]) #drop all columns that are strings or not important
df = df.fillna(df.median(numeric_only=True)) # fill null values with median

X = df.iloc[:,0:41] #set our x to 
y = df.iloc[:,41]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)



rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf.score(X_test, y_test)


print(classification_report(y_test, y_pred)) # 4 percent will result in false positives -> show implications in readme



# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Sort them highest to lowest
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.title("Feature Importances - Kepler Exoplanet Classifier")
plt.tight_layout()
plt.show()


