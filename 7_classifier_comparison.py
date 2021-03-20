# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

# Import data
broadcast = pd.read_csv("XXX/XXX/broadcast_for_models.csv")

# Check correlation between the dependent and all the other explanatory variables
correlation = broadcast.corr(method="pearson")
cor = correlation['effect_classifier_10_perc']

# Choose dependent and independent variables
y = broadcast['effect_classifier_10_perc'] #Can add more here, e.g. relative or 5 min window

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

X_exdefaults = X.drop(['ad_short',
                       'Sunday',
                       'Any other position',
                       'wasmachines',
                       'morning',
                       'public broadcaster',
                       'program_other'
                            ],axis=1)

# Set random seed for reproducibility
np.random.seed(209734)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

## RANDOM FOREST CLASSIFIER #########################################################################################################
rf = RandomForestClassifier()
grid = dict()
grid['n_estimators'] = [1,5,10,20,50,100,500]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=cv, scoring='recall')
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

## ADABOOST CLASSIFIER #########################################################################################################
abc = AdaBoostClassifier(rf)
grid = dict()
grid['n_estimators'] = [100, 500,750,1000]
grid['learning_rate'] = [1.0, 1.5, 1.9, 2.0]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=abc, param_grid=grid, n_jobs=-1, cv=cv, scoring='recall')
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

## kNN #########################################################################################################
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
knn = KNeighborsClassifier()
grid = dict()
grid['n_neighbors'] = [1,2,3,4,6,8]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring='recall')
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

## SVM #########################################################################################################
sv = svm.SVC()
grid = dict()
grid['kernel'] = ['poly','sigmoid','rbf']
grid['C'] = [1,0.8,0.6,0.2]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=sv, param_grid=grid, n_jobs=-1, cv=cv, scoring='recall')
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

accuracies_rf = []
recall_rf = []
precision_rf = []
rf_importances = np.zeros((10,len(X.columns)))

accuracies_abc = []
recall_abc = []
precision_abc = []
abc_importances = np.zeros((10,len(X.columns)))

accuracies_knn = []
recall_knn = []
precision_knn = []

accuracies_lda = []
recall_lda = []
precision_lda = []

accuracies_lr = []
recall_lr = []
precision_lr = []

accuracies_gnb = []
recall_gnb = []
precision_gnb = []

accuracies_sv = []
recall_sv = []
precision_sv = []

# Set random seed for reproducibility
np.random.seed(209734)

i = 0
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    #Select model with best hyperparameters from grid search
    rf=RandomForestClassifier(n_estimators=1)
    rf_fit = rf.fit(X_train, y_train)
    rf_pred = rf_fit.predict(X_test)
    
    rf_importances[i,:] = rf.feature_importances_
    
    test_accuracy_rf = metrics.accuracy_score(y_test, rf_pred)
    test_recall_rf = metrics.recall_score(y_test, rf_pred)
    test_precision_rf = metrics.precision_score(y_test, rf_pred)
    accuracies_rf.append(test_accuracy_rf)
    recall_rf.append(test_accuracy_rf)
    precision_rf.append(test_accuracy_rf)
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)
    abc_fit = abc.fit(X_train, y_train)
    abc_pred = abc_fit.predict(X_test)
    
    abc_importances[i,:] = abc.feature_importances_
    
    test_accuracy_abc = metrics.accuracy_score(y_test, abc_pred)
    test_recall_abc = metrics.recall_score(y_test, abc_pred)
    test_precision_abc = metrics.precision_score(y_test, abc_pred)
    accuracies_abc.append(test_accuracy_abc)
    recall_abc.append(test_accuracy_abc)
    precision_abc.append(test_accuracy_abc)
    
    knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
    knn_fit = knn.fit(X_train, y_train)
    knn_pred = knn_fit.predict(X_test)
    
    test_accuracy_knn = metrics.accuracy_score(y_test, knn_pred)
    test_recall_knn = metrics.recall_score(y_test, knn_pred)
    test_precision_knn = metrics.precision_score(y_test, knn_pred)
    accuracies_knn.append(test_accuracy_knn)
    recall_knn.append(test_accuracy_knn)
    precision_knn.append(test_accuracy_knn)
    
    lda = LinearDiscriminantAnalysis()
    lda_fit = lda.fit(X_train, y_train)
    lda_pred = lda_fit.predict(X_test)
    
    test_accuracy_lda = metrics.accuracy_score(y_test, lda_pred)
    test_recall_lda = metrics.recall_score(y_test, lda_pred)
    test_precision_lda = metrics.precision_score(y_test, lda_pred)
    accuracies_lda.append(test_accuracy_lda)
    recall_lda.append(test_accuracy_lda)
    precision_lda.append(test_accuracy_lda)
    
    X_train_exdefault, X_test_exdefault = X_exdefaults.iloc[train_index], X_exdefaults.iloc[test_index]
            
    lr = LogisticRegression(solver='liblinear', random_state=0)
    lr_fit = lr.fit(X_train_exdefault, y_train)
    lr_pred = lr_fit.predict(X_test_exdefault)
    
    test_accuracy_lr = metrics.accuracy_score(y_test, lr_pred)
    test_recall_lr = metrics.recall_score(y_test, lr_pred)
    test_precision_lr = metrics.precision_score(y_test, lr_pred)
    accuracies_lr.append(test_accuracy_lr)
    recall_lr.append(test_accuracy_lr)
    precision_lr.append(test_accuracy_lr)
    
    gnb = GaussianNB()
    gnb_fit = gnb.fit(X_train, y_train)
    gnb_pred = gnb_fit.predict(X_test)
    
    test_accuracy_gnb = metrics.accuracy_score(y_test, gnb_pred)
    test_recall_gnb = metrics.recall_score(y_test, gnb_pred)
    test_precision_gnb = metrics.precision_score(y_test, gnb_pred)
    accuracies_gnb.append(test_accuracy_gnb)
    recall_gnb.append(test_accuracy_gnb)
    precision_gnb.append(test_accuracy_gnb)
    
    sv = svm.SVC(kernel = 'rbf', C = 1)
    sv_fit = sv.fit(X_train, y_train)
    sv_pred = sv_fit.predict(X_test)
    
    test_accuracy_sv = metrics.accuracy_score(y_test, sv_pred)
    test_recall_sv = metrics.recall_score(y_test, sv_pred)
    test_precision_sv = metrics.precision_score(y_test, sv_pred)
    accuracies_sv.append(test_accuracy_sv)
    recall_sv.append(test_accuracy_sv)
    precision_sv.append(test_accuracy_sv)
    
    i = i + 1
    
    #print("Test accuracy dt: %f" % test_accuracy_dt)

# Report: Table 7
mean_accuracy_rf = test_accuracy_rf.mean()
mean_recall_rf = test_recall_rf.mean()
mean_precision_rf = test_precision_rf.mean()
performances_rf = (mean_accuracy_rf,mean_recall_rf,mean_precision_rf)

mean_accuracy_abc = test_accuracy_abc.mean()
mean_recall_abc = test_recall_abc.mean()
mean_precision_abc = test_precision_abc.mean()
performances_abc = (mean_accuracy_abc,mean_recall_abc,mean_precision_abc)

mean_accuracy_knn = test_accuracy_knn.mean()
mean_recall_knn = test_recall_knn.mean()
mean_precision_knn = test_precision_knn.mean()
performances_knn = (mean_accuracy_knn,mean_recall_knn,mean_precision_knn)

mean_accuracy_lda = test_accuracy_lda.mean()
mean_recall_lda = test_recall_lda.mean()
mean_precision_lda = test_precision_lda.mean()
performances_lda = (mean_accuracy_lda,mean_recall_lda,mean_precision_lda)

mean_accuracy_lr = test_accuracy_lr.mean()
mean_recall_lr = test_recall_lr.mean()
mean_precision_lr = test_precision_lr.mean()
performances_lr = (mean_accuracy_lr,mean_recall_lr,mean_precision_lr)

mean_accuracy_gnb = test_accuracy_gnb.mean()
mean_recall_gnb = test_recall_gnb.mean()
mean_precision_gnb = test_precision_gnb.mean()
performances_gnb = (mean_accuracy_gnb,mean_recall_gnb,mean_precision_gnb)

mean_accuracy_sv = test_accuracy_sv.mean()
mean_recall_sv = test_recall_sv.mean()
mean_precision_sv = test_precision_sv.mean()
performances_sv = (mean_accuracy_sv,mean_recall_sv,mean_precision_sv)

# Report: Table 7
classifier_results = pd.DataFrame(columns=['classifier','rf','abc','knn','lda','lr', 'gnb','sv'])
classifier_results['classifier'] = ['accuracy','recall','precision']
classifier_results['rf'] = performances_rf
classifier_results['abc'] = performances_abc
classifier_results['knn'] = performances_knn
classifier_results['lda'] = performances_lda
classifier_results['lr'] = performances_lr
classifier_results['gnb'] = performances_gnb
classifier_results['sv'] = performances_sv


## Importances RF #########################################################################################################
importances_rf = rf_importances.mean(axis=0)

sorted_indices = np.argsort(importances_rf)[::-1]

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances_rf[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

## Importances ABC #########################################################################################################
importances_abc = abc_importances.mean(axis=0)

sorted_indices = np.argsort(importances_abc)[::-1]

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances_abc[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

importances = list(zip(X.columns[sorted_indices], importances_abc[sorted_indices]))


classifier_results.to_csv('classifier_comparison_results.csv')
