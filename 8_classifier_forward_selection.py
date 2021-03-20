# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import StratifiedKFold

# Import data
broadcast = pd.read_csv("XXX/XXX/broadcast_for_models.csv")

# Define dependent variable
y = broadcast['effect_classifier_10_perc']

# Define independent variables
X = broadcast[['gross_rating_point',
               'ad_long','ad_mid', 'ad_short',
               'between_programs',
               'Monday','Tuesday','Wednesday','Thursday','Friday','Saturay', 'Sunday',
               'First Position', 'Last Position','Second Position', 'Any other position',
               'televisies','laptops', 'wasmachines',
               'night','morning','afternoon','prime','late_evening',
               'weekend',
               #channel groups
               'business',
               'commercial broadcaster', 
               'cooking',
               'drama/crime', 
               'men', 
               'music', 
               'science', 
               'sports', 
               'women', 
               'youth', 
               'public broadcaster',
               #program categories
               'program_kids', 
               'program_news', 
               'program_films', 
               'program_series', 
               'program_sports',
               'program_other'
               ]]

# Set random seed for reproducibility
np.random.seed(209734)

# Initialize 5 folds and the AdaBoostClassifier
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

# Perform forward selection with score = recall and with score = accuracy
sfs_r = sfs(abc,
           k_features=20,
           forward=True,
           floating=False,
           verbose=10,
           scoring='recall',
           cv=5)

sfs_a = sfs(abc,
           k_features=20,
           forward=True,
           floating=False,
           verbose=10,
           scoring='accuracy',
           cv=5)

sfs_r_fit = sfs_r.fit(X, y)
sfs_a_fit = sfs_a.fit(X, y)

# The selected sets of features, cut off when no improvement is made by adding another feature
selected_featuresets_r = sfs_r_fit.subsets_ 
#1: gross rating, 2: laptops, 3: night, 4: weekend, 5: ad_mid, 6: ad_long, 7: "any other position'

selected_featuresets_a = sfs_a_fit.subsets_
#1: gross rating, 2: ad_short

X_r = X[['gross_rating_point','laptops','night','weekend','ad_mid','ad_long','Any other position']]
X_a = X[['gross_rating_point','ad_short']]
X = X[['gross_rating_point','afternoon','commercial broadcaster',
       'prime','morning','public broadcaster',
       'night','wasmachines','weekend','ad_short',
       'laptops','ad_mid','sports','program_sports']]

# Perform 5-fold CV to fit the classifier using feature sets X_r, X_a and X 
# Report the avg accuracy, avg recall, avg precision and avg feature importances

accuracies_sfs_r = []
recalls_sfs_r = []
precisions_sfs_r = []
importances_sfs_r = np.zeros((5,len(X_r.columns)))

accuracies_sfs_a = []
recalls_sfs_a = []
precisions_sfs_a = []
importances_sfs_a = np.zeros((5,len(X_a.columns)))

accuracies = []
recalls = []
precisions = []
importances = np.zeros((5,len(X.columns)))

i=0
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index],X.iloc[test_index]
    X_train_r, X_test_r = X_r.iloc[train_index], X_r.iloc[test_index]
    X_train_a, X_test_a = X_a.iloc[train_index], X_a.iloc[test_index]
    
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit abc with features selected by forward selection using recall
    abc_fit_r = abc.fit(X_train_r, y_train)
    abc_pred_r = abc_fit_r.predict(X_test_r)
    
    importances_sfs_r[i,:] = abc_fit_r.feature_importances_
    accuracies_sfs_r.append(metrics.accuracy_score(y_test, abc_pred_r))
    recalls_sfs_r.append(metrics.recall_score(y_test, abc_pred_r))
    precisions_sfs_r.append(metrics.precision_score(y_test, abc_pred_r))
    
    # Fit abc with features selected by forward selection using accuracy
    abc_fit_a = abc.fit(X_train_a, y_train)
    abc_pred_a = abc_fit_a.predict(X_test_a)
    
    importances_sfs_a[i,:] = abc_fit_a.feature_importances_
    accuracies_sfs_a.append(metrics.accuracy_score(y_test, abc_pred_a))
    recalls_sfs_a.append(metrics.recall_score(y_test, abc_pred_a))
    precisions_sfs_a.append(metrics.precision_score(y_test, abc_pred_a))
    
    # Fit abc with all feaures 
    abc_fit = abc.fit(X_train, y_train)
    abc_pred = abc_fit.predict(X_test)
    
    importances[i,:] = abc_fit.feature_importances_
    accuracies.append(metrics.accuracy_score(y_test, abc_pred))
    recalls.append(metrics.recall_score(y_test, abc_pred))
    precisions.append(metrics.precision_score(y_test, abc_pred))
    
    i = i+1

# Report: Table 8    
# Results forward selection recall
mean_accuracy_sfs_r = np.asarray(accuracies_sfs_r).mean()
mean_recall_sfs_r = np.asarray(recalls_sfs_r).mean()
mean_precision_sfs_r = np.asarray(precisions_sfs_r).mean()
performances_sfs_r = (mean_accuracy_sfs_r,mean_recall_sfs_r,mean_precision_sfs_r)
mean_importances_sfs_r = importances_sfs_r.mean(axis=0)

sorted_indices_r = np.argsort(mean_importances_sfs_r)[::-1]
importances_r = list(zip(X_r.columns[sorted_indices_r], mean_importances_sfs_r[sorted_indices_r]))

# Results forward selection accuracy
mean_accuracy_sfs_a = np.asarray(accuracies_sfs_a).mean()
mean_recall_sfs_a = np.asarray(recalls_sfs_a).mean()
mean_precision_sfs_a = np.asarray(precisions_sfs_a).mean()
performances_sfs_a = (mean_accuracy_sfs_a,mean_recall_sfs_a,mean_precision_sfs_a)
mean_importances_sfs_a = importances_sfs_a.mean(axis=0)

sorted_indices_a = np.argsort(mean_importances_sfs_a)[::-1]
importances_a = list(zip(X_a.columns[sorted_indices_a], mean_importances_sfs_a[sorted_indices_a]))

# Results using all features - extract importances
mean_accuracy = np.asarray(accuracies).mean()
mean_recall = np.asarray(recalls).mean()
mean_precision = np.asarray(precisions).mean()
performances = (mean_accuracy,mean_recall,mean_precision)
mean_importances = importances.mean(axis=0)

sorted_indices = np.argsort(mean_importances)[::-1]
importances = list(zip(X.columns[sorted_indices], mean_importances[sorted_indices]))

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), mean_importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()



