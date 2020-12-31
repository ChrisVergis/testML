#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def one_hot_encode(ds,feature):
    #get DF of dummy variables
    dummies = pd.get_dummies(ds[feature])
    #One dummy variable to drop (Dummy Trap)
    dummyDrop = dummies.columns[0]
    #Create a DF from the original and the dummies' DF
    #Drop the original categorical variable and the one dummy
    final =  pd.concat([ds,dummies], axis='columns').drop([feature,dummyDrop], axis='columns')
    return final

#Get data DF
dataset = pd.read_csv("census_income_dataset.csv")
columns = dataset.columns

#Original dataset has some missing info (coded as "?")
#In case such an instance appears in some feature, we replace it by the most frequent instance of the feature
imp = SimpleImputer(missing_values="?",strategy="most_frequent")
dataset = imp.fit_transform(dataset)
dataset = pd.DataFrame(dataset, columns=columns)

#Perform one-hot-encoding on the DF (See function above) on categorical features
features = ["workclass","marital_status","occupation","relationship","race","sex","native_country"]
for f in features:
    dataset = one_hot_encode(dataset,f)
#Re-order to get ouput feature in last column
dataset = dataset[[c for c in dataset.columns if c!="income_level"]+["income_level"]]
dataset.head()

#Standardize scarse /non boolean Input variables
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
col_names = ['capital_gain','capital_loss']
features = dataset[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

dataset[col_names] = features


# Now we go and take our I/O variables
# Education has been enumerated, so for now use it (so we don't need the "2")
# fnlwgt is unknown so I will not use it
X = dataset.iloc[:,[0]+[i for i in range(3,len(dataset.columns)-1)]].values
Y = dataset.iloc[:,-1].values
le = LabelEncoder()
Y = le.fit_transform(Y)

#Split Train/Test (85%/15%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.15, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[6]:


#Hyperparameter scan
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
n_estimators = [int(x) for x in range(5,21)]
max_features = ['auto','sqrt']
max_depth    = [int(x) for x in range(10,81,10)]+[None]
min_samples_split = [2,5,10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
clas_rf_scan = RandomForestClassifier(criterion='entropy')
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCVestimator = clas_rf_scan, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[7]:


rf_random.best_params_


# In[15]:


from sklearn.metrics import accuracy_score,precision_score,recall_score

def evaluate(model, test_features, test_labels):
    prediction = model.predict(test_features)
    accuracy = 100.*accuracy_score(prediction, test_labels)
    recall   = 100.*recall_score(prediction, test_labels)
    precision= 100.*precision_score(prediction, test_labels)
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('Recall   = {:0.2f}%.'.format(recall))
    print('Precision= {:0.2f}%.'.format(precision))
    return accuracy

base_model = RandomForestClassifier(n_estimators = 15, criterion='entropy', random_state = 0)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# In[ ]:




