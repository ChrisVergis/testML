import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.feature_selection import mutual_info_classif,f_classif,SelectKBest


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

def one_hot_encode(ds,feature):
    #get DF of dummy variables
    dummies = pd.get_dummies(ds[feature])
    #One dummy variable to drop (Dummy Trap)
    dummyDrop = dummies.columns[0]
    #Create a DF from the original and the dummies' DF
    #Drop the original categorical variable and the one dummy
    final =  pd.concat([ds,dummies], axis='columns').drop([feature,dummyDrop], axis='columns')
    return final

def preprocess_data(dataset):
    columns = dataset.columns
    #Original dataset has some missing info (coded as "?")
    #In case such an instance appears in some feature, we replace it by the most frequent instance of the feature
    imp = SimpleImputer(missing_values="?",strategy="most_frequent")
    dataset = imp.fit_transform(dataset)
    dataset = pd.DataFrame(dataset, columns=columns)

    #Perform one-hot-encoding on the DF (See function above) on categorical features
    features = ["workclass","marital_status","occupation","relationship","race","sex","native_country","income_level"]
    for f in features:
        dataset = one_hot_encode(dataset,f)
    #Re-order to get ouput feature in last column
    dataset = dataset[[c for c in dataset.columns if c!=">50K"]+[">50K"]]

    #Standardize scarse /non boolean Input variables
    col_names = ['capital_gain','capital_loss','age']
    features = dataset[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    dataset[col_names] = features

    #Now turn everything into numeric
    for col in dataset.columns:       dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    return dataset

def performRandomGridSearchCV(parameters, criterion='entropy', niter = 100, crossval=3, verbose=1, randstate=0,njob=-1):
    print("Performing RandomGridSearch with CV on RandomForest with criterion: {6} \n=======\n parameters ={0}\nN_iter={1}\nCV={2}\nVerbose={3}\nRandomState={4}\nNjob={5}".format(parameters,niter,crossval,verbose,randstate,njob,criterion))
    clas_rf_scan = RandomForestClassifier(criterion=criterion)
    rf_random = RandomizedSearchCV(
        estimator=clas_rf_scan, 
        param_distributions = parameters, 
        n_iter = niter, 
        cv = crossval, 
        verbose=verbose, 
        random_state=randstate, 
        n_jobs = njob)
    rf_random.fit(X_train, y_train)
    return rf_random

def summary(rf_random):
    base_model = RandomForestClassifier(n_estimators = 15, criterion='entropy', random_state = 0)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)
    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
    print("Results for random tree:\n")
    y_pred_untuned = base_model.predict(X_test)    
    classification_report(y_test, y_pred_untuned)
    disp=plot_confusion_matrix(base_model,X_test,y_test,display_labels=["Rich","Poor"], cmap=plt.cm.Blues)
    disp.ax_.set_title("Untuned RF Classifier")
    
    plt.show()
    print("=================\n")
    print("Results for tuned:\n")
    y_pred_tuned= best_random.predict(X_test)
    classification_report(y_test, y_pred_tuned)
    disp=plot_confusion_matrix(best_random,X_test,y_test,display_labels=["Rich","Poor"], cmap=plt.cm.Blues)
    disp.ax_.set_title("Tuned RGridSearch RF Classifier")
    plt.show()

def perform_feature_Importance(dataset,best_random):
    #Columns of X to rows
    dfcolumns= pd.DataFrame(dataset.iloc[:,0:-1].columns)
    
    #Calculate Score with f_classif
    bestfeatures_fclassif = SelectKBest(score_func=f_classif,k=80)
    fit_fclassif = bestfeatures_fclassif.fit(dataset.iloc[:,0:-1],dataset.iloc[:,-1])
    dfscores_fclassif = pd.DataFrame(fit_fclassif.scores_)

    #Calculate Score with mutualinfoclassif
    bestfeatures_mutualinfoclassif = SelectKBest(score_func=mutual_info_classif,k=80)
    fit_mutualinfoclassif  = bestfeatures_mutualinfoclassif.fit(dataset.iloc[:,0:-1],dataset.iloc[:,-1])
    dfscores_mutualinfoclassif = pd.DataFrame(fit_mutualinfoclassif.scores_)

    #Calculate Score with Feature Importance
    importance = best_random.feature_importances_
    #important_features = []
    #for i,v in enumerate(importance):
    #    if v>0.01:
    #        important_features.append(i)
    #        print('Feature: %0d, Score: %.5f' % (i,v))
    #plt.bar([x for x in range(len(importance))], importance)
    #plt.show()

    dffeature_imp = pd.DataFrame(importance)

    featureImportances = pd.concat([dfcolumns, dfscores_fclassif, dfscores_mutualinfoclassif ,dffeature_imp],axis=1)
    featureImportances.columns=["Feature","f_classif_Score","mutualinfo_classif_Score","feature_importance_Score"]
    return featureImportances
    

def plot_Importances(N_features = 20, ordering='feature_importance_Score'):
    #To make nice comparison plot, rescale the scores in [0-1]
    Scaler =     MinMaxScaler()
    FeaturesPlot=featureImportances.copy()
    columns_to_rescale= FeaturesPlot.columns[1:]
    for col in columns_to_rescale:
        FeaturesPlot[col] = Scaler.fit_transform(featureImportances[col].values.reshape(-1,1))
        
    #Now plot the N_feature top scores of the wanted score (ordering)
    FeaturesPlot=FeaturesPlot.nlargest(N_features,ordering)
    FeaturesPlot.plot.bar('Feature')
    plt.show()

#Get data DF
dataset = pd.read_csv("census_income_dataset.csv")
dataset.drop("education",inplace=True,axis=1)
dataset.drop("fnlwgt",inplace=True,axis=1)

dataset = preprocess_data(dataset)
X = dataset.iloc[:,0:-1].values #[0:-1]+[i for i in range(3,len(dataset.columns)-1)]].values
Y = dataset.iloc[:,-1].values
#Split Train/Test (85%/15%)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.15, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#Hyperparameter scan
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

# Fit the random search model
rf_random = performRandomGridSearchCV(random_grid)
summary(rf_random)
best_random = rf_random.best_estimator_

#Perform Feature Importance
featureImportances= perform_feature_Importance(dataset,best_random)
plot_Importances(N_features = 20, ordering='feature_importance_Score')

#FeaturesPlot.index = dataset.iloc[:-1].columns

#FeaturesPlot.nlargest(15,'feature_importance_Score')
