#modello basato sui fattori ambientali
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('dataset/mainDataset.csv')
df.loc[:, 'Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

Y = df['Heart Attack Risk']
X=df[['Age', 'Gender', 'Smoking', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Medication Use', 'Stress Level', 'BMI', 'Sleep Hours Per Day']]

#applico sia smote che undersampling perchè in passato ha funzionato meglio

from imblearn.over_sampling import SMOTE
#undersampling
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import confusion_matrix, classification_report

smote = SMOTE(random_state=42, sampling_strategy=0.85)
undersampler= RandomUnderSampler(random_state=42, sampling_strategy=0.9)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

X_balanced, Y_balanced = undersampler.fit_resample(X_train_smote, Y_train_smote)

#X_with_Y_b = X_balanced.copy()
#X_with_Y_b['Heart Attack Risk'] = Y_balanced
#
#X_with_Y_b.to_csv('dataset/X_with_Y_b.csv', index=False)

print(Y_train.value_counts())
print(Y_train_smote.value_counts())
print(Y_balanced.value_counts())

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

rf_weighted_best = RandomForestClassifier(
    class_weight={0: 1, 1: 3}, n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42
)
rf_weighted_best.fit(X_train_smote, Y_train_smote)
rf_weighted = RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42)
rf_weighted.fit(X_train_smote, Y_train_smote)

print('Random Forest weighted')
print('train set')
print(confusion_matrix(Y_train_smote, rf_weighted.predict(X_train_smote)))
print(classification_report(Y_train_smote, rf_weighted.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, rf_weighted.predict(X_test)))
print(classification_report(Y_test, rf_weighted.predict(X_test)))

print('Random Forest weighted best')
print('train set')
print(confusion_matrix(Y_train_smote, rf_weighted_best.predict(X_train_smote)))
print(classification_report(Y_train_smote, rf_weighted_best.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, rf_weighted_best.predict(X_test)))
print(classification_report(Y_test, rf_weighted_best.predict(X_test)))

from sklearn.neighbors import KNeighborsClassifier

#param_grid_knn = {
#    'n_neighbors': [1, 3, 5, 10],
#    'weights': ['uniform', 'distance', None],
#    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'hamming']
#}
#knn_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3, n_jobs=-1, verbose=2, scoring='f1')
#knn_grid_search.fit(X_train_smote, Y_train_smote)
#print("Best parameters for KNN found: ", knn_grid_search.best_params_)
#n_neighbors = 1, weights = uniform, metric = euclidean

#knearest classifier

knn = KNeighborsClassifier(n_neighbors=5)
knn_best = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='euclidean')
knn_best_balanced = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='euclidean')
knn.fit(X_train_smote, Y_train_smote)
knn_best.fit(X_train_smote, Y_train_smote)
knn_best_balanced.fit(X_balanced, Y_balanced)

print('KNN')
print('train set')
print(confusion_matrix(Y_train_smote, knn.predict(X_train_smote)))
print(classification_report(Y_train_smote, knn.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, knn.predict(X_test)))
print(classification_report(Y_test, knn.predict(X_test)))

print('KNN best')
print('train set')
print(confusion_matrix(Y_train_smote, knn_best.predict(X_train_smote)))
print(classification_report(Y_train_smote, knn_best.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, knn_best.predict(X_test)))
print(classification_report(Y_test, knn_best.predict(X_test)))

print('KNN best balanced')
print('train set')
print(confusion_matrix(Y_balanced, knn_best_balanced.predict(X_balanced)))
print(classification_report(Y_balanced, knn_best_balanced.predict(X_balanced)))
print('test set')
print(confusion_matrix(Y_test, knn_best_balanced.predict(X_test)))
print(classification_report(Y_test, knn_best_balanced.predict(X_test)))

#bagged trees
from sklearn.ensemble import BaggingClassifier

#ricerca per i parametri migliori per Bagging
#param_grid_bagging = {
#    'n_estimators': [100, 150, 200],
#    'max_samples': [1.0, 1.50, 2.0],
#    'max_features': [1.0, 1.50, 2.0],
#    'bootstrap': [True, False],
#    'bootstrap_features': [True, False]
#}
#bagging_grid_search = GridSearchCV(BaggingClassifier(), param_grid_bagging, cv=3, n_jobs=-1, verbose=2, scoring='f1')
#bagging_grid_search.fit(X_train_smote, Y_train_smote)
#print("Best parameters for Bagging found: ", bagging_grid_search.best_params_)
#bootstrap = False, bootstrap_features = False, max_features = 1.0, max_samples = 1.0, n_estimators = 150


bagging = BaggingClassifier(random_state=42)
bagging_balanced = BaggingClassifier(random_state=42)
bagging_best= BaggingClassifier(
    bootstrap=False, bootstrap_features=False, max_features=0.75, max_samples=0.75, n_estimators=125, random_state=42)
bagging_balanced_best=BaggingClassifier(
    bootstrap=False, bootstrap_features=False, max_features=0.75, max_samples=0.75, n_estimators=125, random_state=42)

bagging.fit(X_train_smote, Y_train_smote)
bagging_balanced.fit(X_balanced, Y_balanced)
bagging_best.fit(X_train_smote, Y_train_smote)
bagging_balanced_best.fit(X_balanced, Y_balanced)

print('Bagging')
print('train set')
print(confusion_matrix(Y_train_smote, bagging.predict(X_train_smote)))
print(classification_report(Y_train_smote, bagging.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, bagging.predict(X_test)))
print(classification_report(Y_test, bagging.predict(X_test)))

print('Bagging balanced')
print('train set')
print(confusion_matrix(Y_balanced, bagging_balanced.predict(X_balanced)))
print(classification_report(Y_balanced, bagging_balanced.predict(X_balanced)))
print('test set')
print(confusion_matrix(Y_test, bagging_balanced.predict(X_test)))
print(classification_report(Y_test, bagging_balanced.predict(X_test)))

print('Bagging best')
print('train set')
print(confusion_matrix(Y_train_smote, bagging_best.predict(X_train_smote)))
print(classification_report(Y_train_smote, bagging_best.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, bagging_best.predict(X_test)))
print(classification_report(Y_test, bagging_best.predict(X_test)))

print('Bagging balanced best')
print('train set')
print(confusion_matrix(Y_balanced, bagging_balanced_best.predict(X_balanced)))
print(classification_report(Y_balanced, bagging_balanced_best.predict(X_balanced)))
print('test set')
print(confusion_matrix(Y_test, bagging_balanced_best.predict(X_test)))
print(classification_report(Y_test, bagging_balanced_best.predict(X_test)))
    


#SVC
#from sklearn.svm import SVC

#ricerca dei parametri migliori per SVC
#param_grid_svc = {
#    'C': [0.1, 1, 10],
#    'kernel': ['linear', 'rbf', 'poly'],
#    'gamma': ['scale', 'auto']
#}
#from sklearn.model_selection import GridSearchCV
#svc_grid_search = GridSearchCV(SVC(), param_grid_svc, cv=3, n_jobs=-1, verbose=2, scoring='f1')
#svc_grid_search.fit(X_train_smote, Y_train_smote)
#print("Best parameters for SVC found: ", svc_grid_search.best_params_)
# C = 10, kernel = rbf, gamma = auto


#SVC sembra essere il peggiore
#svc = SVC(random_state=42, class_weight='balanced', C=10, kernel='rbf', gamma='scale')
#svc.fit(X_train_smote, Y_train_smote)
#
#print('SVC')
#print('train set')
#print(confusion_matrix(Y_train_smote, svc.predict(X_train_smote)))
#print(classification_report(Y_train_smote, svc.predict(X_train_smote)))
#print('test set')
#print(confusion_matrix(Y_test, svc.predict(X_test)))
#print(classification_report(Y_test, svc.predict(X_test)))

#non so che fare, divido il modello per maschi e femmine

#df_male = df[df['Gender'] == 1]
#df_female=df[df['Gender'] == 0]
#
#Y_male = df_male['Heart Attack Risk']
#Y_female = df_female['Heart Attack Risk']
#
#X_male=df_male[['Age', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week' , 'Previous Heart Problems' , 'Medication Use', 'Stress Level' , 'Sedentary Hours Per Day', 'BMI', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
#X_female=df_female[['Age', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week' , 'Previous Heart Problems' , 'Medication Use', 'Stress Level' , 'Sedentary Hours Per Day', 'BMI', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
#
#X_male_train, X_male_test, Y_male_train, Y_male_test = train_test_split(X_male, Y_male, test_size=0.3, random_state=42)
#X_male_train_smote, Y_male_train_smote = smote.fit_resample(X_male_train, Y_male_train)
#
#X_female_train, X_female_test, Y_female_train, Y_female_test = train_test_split(X_female, Y_female, test_size=0.3, random_state=42)
#X_female_train_smote, Y_female_train_smote = smote.fit_resample(X_female_train, Y_female_train)
#
#rf_male = RandomForestClassifier(class_weight={0: 1, 1: 3})
#rf_female= RandomForestClassifier(class_weight={0: 1, 1: 3})
#
#rf_male.fit(X_male_train_smote, Y_male_train_smote)
#rf_female.fit(X_female_train_smote, Y_female_train_smote)
#
#print('Random forest male')
#print('train set')
#print(confusion_matrix(Y_male_train_smote, rf_male.predict(X_male_train_smote)))
#print(classification_report(Y_male_train_smote, rf_male.predict(X_male_train_smote)))
#print('test set')
#print(confusion_matrix(Y_male_test, rf_male.predict(X_male_test)))
#print(classification_report(Y_male_test, rf_male.predict(X_male_test)))
#
#print('Random forest female')
#print('train set')
#print(confusion_matrix(Y_female_train_smote, rf_male.predict(X_female_train_smote)))
#print(classification_report(Y_female_train_smote, rf_male.predict(X_female_train_smote)))
#print('test set')
#print(confusion_matrix(Y_female_test, rf_male.predict(X_female_test)))
#print(classification_report(Y_female_test, rf_male.predict(X_female_test)))
#
#knn_male = KNeighborsClassifier(n_neighbors=5)
#knn_female = KNeighborsClassifier(n_neighbors=5)
#
#knn_male.fit(X_male_train_smote, Y_male_train_smote)
#knn_female.fit(X_female_train_smote, Y_female_train_smote)
#
#print('knn male')
#print('train set')
#print(confusion_matrix(Y_male_train_smote, knn_male.predict(X_male_train_smote)))
#print(classification_report(Y_male_train_smote, knn_male.predict(X_male_train_smote)))
#print('test set')
#print(confusion_matrix(Y_male_test, knn_male.predict(X_male_test)))
#print(classification_report(Y_male_test, knn_male.predict(X_male_test)))
#
#print('knn female')
#print('train set')
#print(confusion_matrix(Y_female_train_smote, knn_female.predict(X_female_train_smote)))
#print(classification_report(Y_female_train_smote, knn_female.predict(X_female_train_smote)))
#print('test set')
#print(confusion_matrix(Y_female_test, knn_female.predict(X_female_test)))
#print(classification_report(Y_female_test, knn_female.predict(X_female_test)))
