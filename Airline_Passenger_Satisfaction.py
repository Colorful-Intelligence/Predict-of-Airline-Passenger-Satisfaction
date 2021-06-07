"""
Airline Passenger Satisfaction
"""
#%% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")

#%% Read the dataset

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data = pd.concat([train,test],axis = 0)

#%% Exploratory Data Analysis (EDA)

data.columns
data.info()
data.describe

data.drop(["Unnamed: 0","id"],axis = 1,inplace = True)

# Target column is satisfication , I'm going to change name of this column as a "Target"
data.rename({"satisfaction":"Target"},axis = 1,inplace = True) # axis = 1 , which means column

data.columns

"""
['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
       'Target'],

"""

#%% Missing Values


data.columns[data.isnull().any()] # 'Arrival Delay in Minutes'
data.isnull().sum() # 'Arrival Delay in Minutes' have 393 missing values

#%% Fill Missing Values
# Here I will intuitively fill in the missing values of the 'Arrival Delay Per Minute' mean.

data["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"].fillna(data["Arrival Delay in Minutes"].mean())

#%% Correlation Matrix
corr_matrix = data.corr()
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr_matrix,annot = True,linewidths=0.5,fmt = ".2f")
plt.title("Correlation Matrix between Features(Columns)")
plt.show()

#%% Outlier Detection

from collections import Counter

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,['Age','Flight Distance','Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'])]
#%% Drop Outliers

data = data.drop(detect_outliers(data,['Age','Flight Distance','Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']),axis = 0).reset_index(drop = True)

#%% Label Encoder Operation

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

columnsToEncode = list(data.select_dtypes(include = ["category","object"]))
for feature in columnsToEncode:
    data[feature] = le.fit_transform(data[feature])

#%% To get X and Y Coordinates

y = data.Target.values
x_data = data.drop(["Target"],axis = 1)

#%% Normalization Operation

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%% Train-Test Split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


#%% Support Vector Machines Classification

from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)



print("Accuracy of the Support Vector Machine : ",svm.score(x_test,y_test)*100)
"""
Accuracy of the Support Vector Machine :  95.01046193465315
"""

#%% Random Forest Classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print("Accuracy of the Random Forest Classification : % {}".format(accuracy_score(y_test,predicted)*100))

"""
Accuracy of the Random Forest Classification : % 96.23370352486721
"""

#%% K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = rf, X = x_train,y = y_train,cv = 10)

print("average accuracy = ",np.mean(accuracies)) # average accuracy =  0.9625777418762652
print("average std = ",np.std(accuracies)) # average std =  0.0021439007198008923

#%% Confusion Matrix

y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor = "red",fmt = ".0f")
plt.ylabel("y_pred")
plt.xlabel("y_true")
plt.title("Confusion Matrix")
plt.show()

