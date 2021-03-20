# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

# Import data
broadcast = pd.read_csv("XXX/XXX/broadcast_for_models.csv")

# Choose dependent and independent variables
y_12 = broadcast['effect_classifier_12_perc']
y_10 = broadcast['effect_classifier_10_perc'] 
y_8 = broadcast['effect_classifier_8_perc']
y_5 = broadcast['effect_classifier_5_perc']

X_forward_recall = broadcast[['gross_rating_point',
               'ad_long',
               'televisies','laptops', 
               'night',
               ]]

X_forward_precision = broadcast[['gross_rating_point','ad_short']]

X_feature_importance = broadcast[['gross_rating_point',
               'night',
               'morning',
               'afternoon',
               'prime',
               'commercial broadcaster', 
               'cooking',
               'public broadcaster',
               'wasmachines'
               ]]

# Set random seed for reproducibility
np.random.seed(209734)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracies_abc_12_forward_recall = []
recall_abc_12_forward_recall = []
precision_abc_12_forward_recall= []

accuracies_abc_10_forward_recall = []
recall_abc_10_forward_recall = []
precision_abc_10_forward_recall= []

accuracies_abc_8_forward_recall = []
recall_abc_8_forward_recall = []
precision_abc_8_forward_recall= []

accuracies_abc_5_forward_recall = []
recall_abc_5_forward_recall = []
precision_abc_5_forward_recall= []

#FORWARD SELECTION RECALL
#12%

for train_index, test_index in kf.split(X_forward_recall,y_12):
    X_forward_recall_train, X_forward_recall_test = X_forward_recall.iloc[train_index], X_forward_recall.iloc[test_index]
    y_12_train, y_12_test = y_12.iloc[train_index], y_12.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)
    abc_12_forward_recall_fit = abc.fit( X_forward_recall_train, y_12_train)
    abc_12_forward_recall_pred = abc_12_forward_recall_fit.predict(X_forward_recall_test)

    test_accuracy_abc_12_forward_recall = metrics.accuracy_score(y_12_test, abc_12_forward_recall_pred)
    test_recall_abc_12_forward_recall = metrics.recall_score(y_12_test, abc_12_forward_recall_pred)
    test_precision_abc_12_forward_recall = metrics.precision_score(y_12_test, abc_12_forward_recall_pred)
    accuracies_abc_12_forward_recall.append(test_accuracy_abc_12_forward_recall)
    recall_abc_12_forward_recall.append(test_accuracy_abc_12_forward_recall)
    precision_abc_12_forward_recall.append(test_accuracy_abc_12_forward_recall)
    
mean_accuracy_abc_12_forward_recall = test_accuracy_abc_12_forward_recall.mean()
mean_recall_abc_12_forward_recall = test_recall_abc_12_forward_recall.mean()
mean_precision_abc_12_forward_recall = test_precision_abc_12_forward_recall.mean()
performances_abc_12_forward_recall = (mean_accuracy_abc_12_forward_recall,mean_recall_abc_12_forward_recall,mean_precision_abc_12_forward_recall)

#10%

for train_index, test_index in kf.split(X_forward_recall,y_10):
    X_forward_recall_train, X_forward_recall_test = X_forward_recall.iloc[train_index], X_forward_recall.iloc[test_index]
    y_10_train, y_10_test = y_10.iloc[train_index], y_10.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_10_forward_recall_fit = abc.fit( X_forward_recall_train, y_10_train)
    abc_10_forward_recall_pred = abc_10_forward_recall_fit.predict(X_forward_recall_test)

    test_accuracy_abc_10_forward_recall = metrics.accuracy_score(y_10_test, abc_10_forward_recall_pred)
    test_recall_abc_10_forward_recall = metrics.recall_score(y_10_test, abc_10_forward_recall_pred)
    test_precision_abc_10_forward_recall = metrics.precision_score(y_10_test, abc_10_forward_recall_pred)
    accuracies_abc_10_forward_recall.append(test_accuracy_abc_10_forward_recall)
    recall_abc_10_forward_recall.append(test_accuracy_abc_10_forward_recall)
    precision_abc_10_forward_recall.append(test_accuracy_abc_10_forward_recall)
    
mean_accuracy_abc_10_forward_recall = test_accuracy_abc_10_forward_recall.mean()
mean_recall_abc_10_forward_recall = test_recall_abc_10_forward_recall.mean()
mean_precision_abc_10_forward_recall = test_precision_abc_10_forward_recall.mean()
performances_abc_10_forward_recall = (mean_accuracy_abc_10_forward_recall,mean_recall_abc_10_forward_recall,mean_precision_abc_10_forward_recall)

#8%

for train_index, test_index in kf.split(X_forward_recall,y_8):
    X_forward_recall_train, X_forward_recall_test = X_forward_recall.iloc[train_index], X_forward_recall.iloc[test_index]
    y_8_train, y_8_test = y_8.iloc[train_index], y_8.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_8_forward_recall_fit = abc.fit( X_forward_recall_train, y_8_train)
    abc_8_forward_recall_pred = abc_8_forward_recall_fit.predict(X_forward_recall_test)

    test_accuracy_abc_8_forward_recall = metrics.accuracy_score(y_8_test, abc_8_forward_recall_pred)
    test_recall_abc_8_forward_recall = metrics.recall_score(y_8_test, abc_8_forward_recall_pred)
    test_precision_abc_8_forward_recall = metrics.precision_score(y_8_test, abc_8_forward_recall_pred)
    accuracies_abc_8_forward_recall.append(test_accuracy_abc_8_forward_recall)
    recall_abc_8_forward_recall.append(test_accuracy_abc_8_forward_recall)
    precision_abc_8_forward_recall.append(test_accuracy_abc_8_forward_recall)
    
mean_accuracy_abc_8_forward_recall = test_accuracy_abc_8_forward_recall.mean()
mean_recall_abc_8_forward_recall = test_recall_abc_8_forward_recall.mean()
mean_precision_abc_8_forward_recall = test_precision_abc_8_forward_recall.mean()
performances_abc_8_forward_recall = (mean_accuracy_abc_8_forward_recall,mean_recall_abc_8_forward_recall,mean_precision_abc_8_forward_recall)

#5%

for train_index, test_index in kf.split(X_forward_recall,y_5):
    X_forward_recall_train, X_forward_recall_test = X_forward_recall.iloc[train_index], X_forward_recall.iloc[test_index]
    y_5_train, y_5_test = y_5.iloc[train_index], y_5.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_5_forward_recall_fit = abc.fit( X_forward_recall_train, y_5_train)
    abc_5_forward_recall_pred = abc_5_forward_recall_fit.predict(X_forward_recall_test)

    test_accuracy_abc_5_forward_recall = metrics.accuracy_score(y_5_test, abc_5_forward_recall_pred)
    test_recall_abc_5_forward_recall = metrics.recall_score(y_5_test, abc_5_forward_recall_pred)
    test_precision_abc_5_forward_recall = metrics.precision_score(y_5_test, abc_5_forward_recall_pred)
    accuracies_abc_5_forward_recall.append(test_accuracy_abc_5_forward_recall)
    recall_abc_5_forward_recall.append(test_accuracy_abc_5_forward_recall)
    precision_abc_5_forward_recall.append(test_accuracy_abc_5_forward_recall)
    
mean_accuracy_abc_5_forward_recall = test_accuracy_abc_5_forward_recall.mean()
mean_recall_abc_5_forward_recall = test_recall_abc_5_forward_recall.mean()
mean_precision_abc_5_forward_recall = test_precision_abc_5_forward_recall.mean()
performances_abc_5_forward_recall = (mean_accuracy_abc_5_forward_recall,mean_recall_abc_5_forward_recall,mean_precision_abc_5_forward_recall)

accuracies_abc_12_forward_precision = []
recall_abc_12_forward_precision = []
precision_abc_12_forward_precision= []

accuracies_abc_10_forward_precision = []
recall_abc_10_forward_precision = []
precision_abc_10_forward_precision= []

accuracies_abc_8_forward_precision = []
recall_abc_8_forward_precision = []
precision_abc_8_forward_precision= []

accuracies_abc_5_forward_precision = []
recall_abc_5_forward_precision = []
precision_abc_5_forward_precision= []

#FORWARD SELECTION PRECISION
#12%

for train_index, test_index in kf.split(X_forward_precision,y_12):
    X_forward_precision_train, X_forward_precision_test = X_forward_precision.iloc[train_index], X_forward_precision.iloc[test_index]
    y_12_train, y_12_test = y_12.iloc[train_index], y_12.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)
    abc_12_forward_precision_fit = abc.fit( X_forward_precision_train, y_12_train)
    abc_12_forward_precision_pred = abc_12_forward_precision_fit.predict(X_forward_precision_test)

    test_accuracy_abc_12_forward_precision = metrics.accuracy_score(y_12_test, abc_12_forward_precision_pred)
    test_recall_abc_12_forward_precision = metrics.recall_score(y_12_test, abc_12_forward_precision_pred)
    test_precision_abc_12_forward_precision = metrics.precision_score(y_12_test, abc_12_forward_precision_pred)
    accuracies_abc_12_forward_precision.append(test_accuracy_abc_12_forward_precision)
    recall_abc_12_forward_precision.append(test_recall_abc_12_forward_precision)
    precision_abc_12_forward_precision.append(test_precision_abc_12_forward_precision)
    
mean_accuracy_abc_12_forward_precision = test_accuracy_abc_12_forward_precision.mean()
mean_recall_abc_12_forward_precision = test_recall_abc_12_forward_precision.mean()
mean_precision_abc_12_forward_precision = test_precision_abc_12_forward_precision.mean()
performances_abc_12_forward_precision = (mean_accuracy_abc_12_forward_precision,mean_recall_abc_12_forward_precision,mean_precision_abc_12_forward_precision)

#10%

for train_index, test_index in kf.split(X_forward_precision,y_10):
    X_forward_precision_train, X_forward_precision_test = X_forward_precision.iloc[train_index], X_forward_precision.iloc[test_index]
    y_10_train, y_10_test = y_10.iloc[train_index], y_10.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_10_forward_precision_fit = abc.fit( X_forward_precision_train, y_10_train)
    abc_10_forward_precision_pred = abc_10_forward_precision_fit.predict(X_forward_precision_test)

    test_accuracy_abc_10_forward_precision = metrics.accuracy_score(y_10_test, abc_10_forward_precision_pred)
    test_recall_abc_10_forward_precision = metrics.recall_score(y_10_test, abc_10_forward_precision_pred)
    test_precision_abc_10_forward_precision = metrics.precision_score(y_10_test, abc_10_forward_precision_pred)
    accuracies_abc_10_forward_precision.append(test_accuracy_abc_10_forward_precision)
    recall_abc_10_forward_precision.append(test_recall_abc_10_forward_precision)
    precision_abc_10_forward_precision.append(test_precision_abc_10_forward_precision)
    
mean_accuracy_abc_10_forward_precision = test_accuracy_abc_10_forward_precision.mean()
mean_recall_abc_10_forward_precision = test_recall_abc_10_forward_precision.mean()
mean_precision_abc_10_forward_precision = test_precision_abc_10_forward_precision.mean()
performances_abc_10_forward_precision = (mean_accuracy_abc_10_forward_precision,mean_recall_abc_10_forward_precision,mean_precision_abc_10_forward_precision)

#8%

for train_index, test_index in kf.split(X_forward_precision,y_8):
    X_forward_precision_train, X_forward_precision_test = X_forward_precision.iloc[train_index], X_forward_precision.iloc[test_index]
    y_8_train, y_8_test = y_8.iloc[train_index], y_8.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_8_forward_precision_fit = abc.fit( X_forward_precision_train, y_8_train)
    abc_8_forward_precision_pred = abc_8_forward_precision_fit.predict(X_forward_precision_test)

    test_accuracy_abc_8_forward_precision = metrics.accuracy_score(y_8_test, abc_8_forward_precision_pred)
    test_recall_abc_8_forward_precision = metrics.recall_score(y_8_test, abc_8_forward_precision_pred)
    test_precision_abc_8_forward_precision = metrics.precision_score(y_8_test, abc_8_forward_precision_pred)
    accuracies_abc_8_forward_precision.append(test_accuracy_abc_8_forward_precision)
    recall_abc_8_forward_precision.append(test_accuracy_abc_8_forward_precision)
    precision_abc_8_forward_precision.append(test_accuracy_abc_8_forward_precision)
    
mean_accuracy_abc_8_forward_precision = test_accuracy_abc_8_forward_precision.mean()
mean_recall_abc_8_forward_precision = test_recall_abc_8_forward_precision.mean()
mean_precision_abc_8_forward_precision = test_precision_abc_8_forward_precision.mean()
performances_abc_8_forward_precision = (mean_accuracy_abc_8_forward_precision,mean_recall_abc_8_forward_precision,mean_precision_abc_8_forward_precision)

#5%

for train_index, test_index in kf.split(X_forward_precision,y_5):
    X_forward_precision_train, X_forward_precision_test = X_forward_precision.iloc[train_index], X_forward_precision.iloc[test_index]
    y_5_train, y_5_test = y_5.iloc[train_index], y_5.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_5_forward_precision_fit = abc.fit( X_forward_precision_train, y_5_train)
    abc_5_forward_precision_pred = abc_5_forward_precision_fit.predict(X_forward_precision_test)

    test_accuracy_abc_5_forward_precision = metrics.accuracy_score(y_5_test, abc_5_forward_precision_pred)
    test_recall_abc_5_forward_precision = metrics.recall_score(y_5_test, abc_5_forward_precision_pred)
    test_precision_abc_5_forward_precision = metrics.precision_score(y_5_test, abc_5_forward_precision_pred)
    accuracies_abc_5_forward_precision.append(test_accuracy_abc_5_forward_precision)
    recall_abc_5_forward_precision.append(test_accuracy_abc_5_forward_precision)
    precision_abc_5_forward_precision.append(test_accuracy_abc_5_forward_precision)
    
mean_accuracy_abc_5_forward_precision = test_accuracy_abc_5_forward_precision.mean()
mean_recall_abc_5_forward_precision = test_recall_abc_5_forward_precision.mean()
mean_precision_abc_5_forward_precision = test_precision_abc_5_forward_precision.mean()
performances_abc_5_forward_precision = (mean_accuracy_abc_5_forward_precision,mean_recall_abc_5_forward_precision,mean_precision_abc_5_forward_precision)

accuracies_abc_12_feature_importance = []
recall_abc_12_feature_importance = []
precision_abc_12_feature_importance= []

accuracies_abc_10_feature_importance = []
recall_abc_10_feature_importance = []
precision_abc_10_feature_importance= []

accuracies_abc_8_feature_importance = []
recall_abc_8_feature_importance = []
precision_abc_8_feature_importance= []

accuracies_abc_5_feature_importance = []
recall_abc_5_feature_importance = []
precision_abc_5_feature_importance= []

#FEATURE IMPORTANCE
#12%

for train_index, test_index in kf.split(X_feature_importance,y_12):
    X_feature_importance_train, X_feature_importance_test = X_feature_importance.iloc[train_index], X_feature_importance.iloc[test_index]
    y_12_train, y_12_test = y_12.iloc[train_index], y_12.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)
    abc_12_feature_importance_fit = abc.fit( X_feature_importance_train, y_12_train)
    abc_12_feature_importance_pred = abc_12_feature_importance_fit.predict(X_feature_importance_test)

    test_accuracy_abc_12_feature_importance = metrics.accuracy_score(y_12_test, abc_12_feature_importance_pred)
    test_recall_abc_12_feature_importance = metrics.recall_score(y_12_test, abc_12_feature_importance_pred)
    test_precision_abc_12_feature_importance = metrics.precision_score(y_12_test, abc_12_feature_importance_pred)
    accuracies_abc_12_feature_importance.append(test_accuracy_abc_12_feature_importance)
    recall_abc_12_feature_importance.append(test_recall_abc_12_feature_importance)
    precision_abc_12_feature_importance.append(test_precision_abc_12_feature_importance)
    
mean_accuracy_abc_12_feature_importance = test_accuracy_abc_12_feature_importance.mean()
mean_recall_abc_12_feature_importance = test_recall_abc_12_feature_importance.mean()
mean_precision_abc_12_feature_importance = test_precision_abc_12_feature_importance.mean()
performances_abc_12_feature_importance = (mean_accuracy_abc_12_feature_importance,mean_recall_abc_12_feature_importance,mean_precision_abc_12_feature_importance)

#10%

for train_index, test_index in kf.split(X_feature_importance,y_10):
    X_feature_importance_train, X_feature_importance_test = X_feature_importance.iloc[train_index], X_feature_importance.iloc[test_index]
    y_10_train, y_10_test = y_10.iloc[train_index], y_10.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_10_feature_importance_fit = abc.fit( X_feature_importance_train, y_10_train)
    abc_10_feature_importance_pred = abc_10_feature_importance_fit.predict(X_feature_importance_test)

    test_accuracy_abc_10_feature_importance = metrics.accuracy_score(y_10_test, abc_10_feature_importance_pred)
    test_recall_abc_10_feature_importance = metrics.recall_score(y_10_test, abc_10_feature_importance_pred)
    test_precision_abc_10_feature_importance = metrics.precision_score(y_10_test, abc_10_feature_importance_pred)
    accuracies_abc_10_feature_importance.append(test_accuracy_abc_10_feature_importance)
    recall_abc_10_feature_importance.append(test_recall_abc_10_feature_importance)
    precision_abc_10_feature_importance.append(test_precision_abc_10_feature_importance)
    
mean_accuracy_abc_10_feature_importance = test_accuracy_abc_10_feature_importance.mean()
mean_recall_abc_10_feature_importance = test_recall_abc_10_feature_importance.mean()
mean_precision_abc_10_feature_importance = test_precision_abc_10_feature_importance.mean()
performances_abc_10_feature_importance = (mean_accuracy_abc_10_feature_importance,mean_recall_abc_10_feature_importance,mean_precision_abc_10_feature_importance)

#8%

for train_index, test_index in kf.split(X_feature_importance,y_8):
    X_feature_importance_train, X_feature_importance_test = X_feature_importance.iloc[train_index], X_feature_importance.iloc[test_index]
    y_8_train, y_8_test = y_8.iloc[train_index], y_8.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_8_feature_importance_fit = abc.fit( X_feature_importance_train, y_8_train)
    abc_8_feature_importance_pred = abc_8_feature_importance_fit.predict(X_feature_importance_test)

    test_accuracy_abc_8_feature_importance = metrics.accuracy_score(y_8_test, abc_8_feature_importance_pred)
    test_recall_abc_8_feature_importance = metrics.recall_score(y_8_test, abc_8_feature_importance_pred)
    test_precision_abc_8_feature_importance = metrics.precision_score(y_8_test, abc_8_feature_importance_pred)
    accuracies_abc_8_feature_importance.append(test_accuracy_abc_8_feature_importance)
    recall_abc_8_feature_importance.append(test_accuracy_abc_8_feature_importance)
    precision_abc_8_feature_importance.append(test_accuracy_abc_8_feature_importance)
    
mean_accuracy_abc_8_feature_importance = test_accuracy_abc_8_feature_importance.mean()
mean_recall_abc_8_feature_importance = test_recall_abc_8_feature_importance.mean()
mean_precision_abc_8_feature_importance = test_precision_abc_8_feature_importance.mean()
performances_abc_8_feature_importance = (mean_accuracy_abc_8_feature_importance,mean_recall_abc_8_feature_importance,mean_precision_abc_8_feature_importance)

#5%

for train_index, test_index in kf.split(X_feature_importance,y_5):
    X_feature_importance_train, X_feature_importance_test = X_feature_importance.iloc[train_index], X_feature_importance.iloc[test_index]
    y_5_train, y_5_test = y_5.iloc[train_index], y_5.iloc[test_index]
    
    abc = AdaBoostClassifier(n_estimators = 100, learning_rate=1.9)

    abc_5_feature_importance_fit = abc.fit( X_feature_importance_train, y_5_train)
    abc_5_feature_importance_pred = abc_5_feature_importance_fit.predict(X_feature_importance_test)

    test_accuracy_abc_5_feature_importance = metrics.accuracy_score(y_5_test, abc_5_feature_importance_pred)
    test_recall_abc_5_feature_importance = metrics.recall_score(y_5_test, abc_5_feature_importance_pred)
    test_precision_abc_5_feature_importance = metrics.precision_score(y_5_test, abc_5_feature_importance_pred)
    accuracies_abc_5_feature_importance.append(test_accuracy_abc_5_feature_importance)
    recall_abc_5_feature_importance.append(test_accuracy_abc_5_feature_importance)
    precision_abc_5_feature_importance.append(test_accuracy_abc_5_feature_importance)
    
mean_accuracy_abc_5_feature_importance = test_accuracy_abc_5_feature_importance.mean()
mean_recall_abc_5_feature_importance = test_recall_abc_5_feature_importance.mean()
mean_precision_abc_5_feature_importance = test_precision_abc_5_feature_importance.mean()
performances_abc_5_feature_importance = (mean_accuracy_abc_5_feature_importance,mean_recall_abc_5_feature_importance,mean_precision_abc_5_feature_importance)

# Report: Table 9
classifier_results_sensitivity_analysis = pd.DataFrame(columns=['classifier','12_forward_recall','10_forward_recall','8_forward_recall','5_forward_recall','12_forward_precision','10_forward_precision','8_forward_precision','5_forward_precision','12_feature_importance','10_feature_importance','8_feature_importance','5_feature_importance'])
classifier_results_sensitivity_analysis['classifier'] = ['accuracy','recall','precision']
classifier_results_sensitivity_analysis['12_forward_recall'] = performances_abc_12_forward_recall
classifier_results_sensitivity_analysis['10_forward_recall'] = performances_abc_10_forward_recall
classifier_results_sensitivity_analysis['8_forward_recall'] = performances_abc_8_forward_recall
classifier_results_sensitivity_analysis['5_forward_recall'] = performances_abc_5_forward_recall
classifier_results_sensitivity_analysis['12_forward_precision'] = performances_abc_12_forward_precision
classifier_results_sensitivity_analysis['10_forward_precision'] = performances_abc_10_forward_precision
classifier_results_sensitivity_analysis['8_forward_precision'] = performances_abc_8_forward_precision
classifier_results_sensitivity_analysis['5_forward_precision'] = performances_abc_5_forward_precision
classifier_results_sensitivity_analysis['12_feature_importance'] = performances_abc_12_feature_importance
classifier_results_sensitivity_analysis['10_feature_importance'] = performances_abc_10_feature_importance
classifier_results_sensitivity_analysis['8_feature_importance'] = performances_abc_8_feature_importance
classifier_results_sensitivity_analysis['5_feature_importance'] = performances_abc_5_feature_importance
















