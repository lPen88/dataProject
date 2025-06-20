#modello basato sui fattori ambientali
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('dataset/mainDatasetAgeFix.csv')
#provo a droppare le righe con ore di sonno 0 perchè i modelli di matlab uscivano meglio
df = df[df['Sleep Hours Per Day'] != 0]
df.loc[:, 'Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

Y = df['Heart Attack Risk']
X=df[['Age', 'Gender', 'Smoking', 'Medication Use', 'Sleep Hours Per Day']]

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, classification_report

smote = SMOTE(random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

#X_with_Y_b = X_balanced.copy()
#X_with_Y_b['Heart Attack Risk'] = Y_balanced
#
#X_with_Y_b.to_csv('dataset/X_with_Y_b.csv', index=False)

print(Y_train.value_counts())
print(Y_train_smote.value_counts())

#param_grid = {
#    'n_estimators': [50, 100, 200],
#    'max_depth': [None, 10, 20, 30],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4]
#}
#rf = RandomForestClassifier(random_state=42)
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
#grid_search.fit(X_train_smote, Y_train_smote)
#print("Best parameters found: ", grid_search.best_params_)
#max_depth = none, min_samples_split = 2, n_estimators = 300, min_samples_leaf = 1

rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=42)
rf.fit(X_train_smote, Y_train_smote)

print('Random Forest weighted')
#print('train set')
#print(confusion_matrix(Y_train_smote, rf_weighted.predict(X_train_smote)))
#print(classification_report(Y_train_smote, rf_weighted.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, rf.predict(X_test)))
print(classification_report(Y_test, rf.predict(X_test)))

from skopt import BayesSearchCV

# 'max_depth': 47, 'min_samples_leaf': 1, 'min_samples_split': 12, 'n_estimators': 300
#bayesian optimization
#param_space = {
#    'n_estimators': (50, 300),
#    'max_depth': (5, 50),     
#    'min_samples_split': (2, 20),
#    'min_samples_leaf': (1, 20), 
#}
#
#
#rf = RandomForestClassifier(random_state=42)
#
#
#bayes_search = BayesSearchCV(
#    estimator=rf,
#    search_spaces=param_space,
#    n_iter=50,
#    cv=5,  
#    random_state=42,
#    n_jobs=-1  # usa tutti i core disponibili
#)
#
#
#bayes_search.fit(X_train_smote, Y_train_smote)
#
#
#print("Best Parameters:", bayes_search.best_params_)
#print("Best Score:", bayes_search.best_score_)

#from sklearn.model_selection import learning_curve
#import matplotlib.pyplot as plt
#import numpy as np
#
#def plot_learning_curve(estimator, X, y, title):
#    train_sizes, train_scores, test_scores = learning_curve(
#        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
#        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
#    )
#    train_scores_mean = np.mean(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#
#    plt.figure(figsize=(7,5))
#    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
#    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
#    plt.title(title)
#    plt.xlabel('Training examples')
#    plt.ylabel('Score')
#    plt.legend(loc='best')
#    plt.grid()
#    plt.show()
#
#plot_learning_curve(rf, X_train_smote, Y_train_smote, "Random Forest (Best Params)")

#from sklearn.ensemble import BaggingClassifier
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
#
#estimator_range = [2,4,6,8,10,12,14,16]
#
#models = []
#scores = []
#
#for n_estimators in estimator_range:
#
#    # Create bagging classifier
#    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)
#
#    # Fit the model
#    clf.fit(X_train_smote, Y_train_smote)
#
#    # Append the model and score to their respective list
#    models.append(clf)
#    scores.append(accuracy_score(y_true = Y_test, y_pred = clf.predict(X_test)))
#
## Generate the plot of scores against number of estimators
#plt.figure(figsize=(9,6))
#plt.plot(estimator_range, scores)
#
## Adjust labels and font (to make visable)
#plt.xlabel("n_estimators", fontsize = 18)
#plt.ylabel("score", fontsize = 18)
#plt.tick_params(labelsize = 16)
#
## Visualize plot
#plt.show() 