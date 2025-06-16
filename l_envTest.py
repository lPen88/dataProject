#modello basato sui fattori ambientali
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('dataset/dataset_l_fix.csv')

df = df[df['Sleep Hours Per Day'] != 0]
df.loc[:, 'Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

Y = df['Heart Attack Risk']
X=df[['Age', 'Sex', 'Cholesterol', 'Systolic', 'Diastolic', 'Heart Rate']]

#plot distribution of age
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#plt.figure(figsize=(10, 6))
#sns.histplot(X['Age'], bins=30, kde=True)
#plt.title('Age Distribution')
#plt.xlabel('Age')
#plt.ylabel('Frequency')
#plt.show()

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, classification_report

smote = SMOTE(random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
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
#'max_depth': 37, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 213

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, Y_train_smote)

print('Random Forest weighted')
#print('train set')
#print(confusion_matrix(Y_train_smote, rf_weighted.predict(X_train_smote)))
#print(classification_report(Y_train_smote, rf_weighted.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, rf.predict(X_test)))
print(classification_report(Y_test, rf.predict(X_test)))

#from skopt import BayesSearchCV
#
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
#    scoring='accuracy',
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


df=pd.read_csv('dataset/test.csv')
Y= df['target']
X=df.drop(columns=['target'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_smote, Y_smote = smote.fit_resample(X_train, Y_train)

rf.fit(X_smote, Y_smote)

print('Test set performance on new data')
print(confusion_matrix(Y_test, rf.predict(X_test)))
print(classification_report(Y_test, rf.predict(X_test)))

