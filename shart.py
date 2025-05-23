#qui addestro i modelli finali
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('dataset/mainDataset.csv')

Y = df['Heart Attack Risk']

X = df.drop(columns=['Heart Attack Risk'])
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

print("\nY training set:\n")
print(Y_train.value_counts())
print("\nAfter applying SMOTE:\n")
print(Y_train_smote.value_counts())

#il codice sotto è per trovare la combinazione migliore di parametri per il random forest
#l'ho commentato perchè ci mette tempo a girare
#una volta runnato mi ha dato come migliore combinazione
#n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2

#param_grid = {
#    'n_estimators': [50, 100, 200],
#    'max_depth': [3, 5, 10, None],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4]
#}
#gs_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
#gs_rf.fit(X_train_smote, Y_train_smote)
#print('Best parameters found by GridSearchCV:', gs_rf.best_params_)

rf_def_smote = RandomForestClassifier(random_state=42)
rf_hyper_smote = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=42)

rf_def_smote.fit(X_train_smote, Y_train_smote)
rf_hyper_smote.fit(X_train_smote, Y_train_smote)

Y_pred_rf_def_smote_train = rf_def_smote.predict(X_train_smote)
Y_pred_rf_def_smote_test = rf_def_smote.predict(X_test)

Y_pred_rf_h_smote_train = rf_hyper_smote.predict(X_train_smote)
Y_pred_rf_h_smote_test = rf_hyper_smote.predict(X_test)

print("\n\nRandom Forest Classifier (default hyperparamteres) on training set with SMOTE\n")
print(confusion_matrix(Y_train_smote, Y_pred_rf_def_smote_train))
print(classification_report(Y_train_smote, Y_pred_rf_def_smote_train))

print("\n\nRandom Forest Classifier (default hyperparamteres) on test set with SMOTE\n")
print(confusion_matrix(Y_test, Y_pred_rf_def_smote_test))
print(classification_report(Y_test, Y_pred_rf_def_smote_test))

print("\n\nRandom Forest Classifier (hyperparamteres) on training set with SMOTE\n")
print(confusion_matrix(Y_train_smote, Y_pred_rf_h_smote_train))
print(classification_report(Y_train_smote, Y_pred_rf_h_smote_train))

print("\n\nRandom Forest Classifier (hyperparamteres) on test set with SMOTE\n")
print(confusion_matrix(Y_test, Y_pred_rf_h_smote_test))
print(classification_report(Y_test, Y_pred_rf_h_smote_test))


##########################################################
##################  FATTORI AMBIENTALI  ##################
##########################################################

print("modello basato su fattori ambientali")

X_env=df[['Age', 'Gender', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week' , 'Previous Heart Problems' , 'Medication Use', 'Stress Level' , 'Sedentary Hours Per Day', 'BMI', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
Y_env = df['Heart Attack Risk']
X_env['Gender'] = X_env['Gender'].map({'Male': 1, 'Female': 0})

X_env_train, X_env_test, Y_env_train, Y_env_test = train_test_split(X_env, Y_env, test_size=0.2, random_state=42)
X_env_train_smote, Y_env_train_smote = smote.fit_resample(X_env_train, Y_env_train)

print("\nY training set:\n")
print(Y_env_train.value_counts())
print("\nAfter applying SMOTE:\n")
print(Y_env_train_smote.value_counts())

#print("ricerca della combinazione migliore di parametri per il random forest")
#param_grid = {
#    'n_estimators': [50, 100, 200],
#    'max_depth': [3, 5, 10, None],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4]
#}
#gs_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
#gs_rf.fit(X_env_train_smote, Y_env_train_smote)
#print('Best parameters found by GridSearchCV:', gs_rf.best_params_)

#qui mi ha dato invece
#n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1

rf_def_env_smote = RandomForestClassifier(random_state=42)
rf_hyper_env_smote = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1, random_state=42)
rf_def_env_smote.fit(X_env_train_smote, Y_env_train_smote)
rf_hyper_env_smote.fit(X_env_train_smote, Y_env_train_smote)

Y_pred_rf_def_env_smote_train = rf_def_env_smote.predict(X_env_train_smote)
Y_pred_rf_def_env_smote_test = rf_def_env_smote.predict(X_env_test)

Y_pred_rf_h_env_smote_train = rf_hyper_env_smote.predict(X_env_train_smote)
Y_pred_rf_h_env_smote_test = rf_hyper_env_smote.predict(X_env_test)

print("\n\nRandom Forest Classifier (default hyperparamteres) on training set with SMOTE (env)\n")
print(confusion_matrix(Y_env_train_smote, Y_pred_rf_def_env_smote_train))
print(classification_report(Y_env_train_smote, Y_pred_rf_def_env_smote_train))

#classe 0 73% classe 1 33%
print("\n\nRandom Forest Classifier (default hyperparamteres) on test set with SMOTE (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_def_env_smote_test))
print(classification_report(Y_env_test, Y_pred_rf_def_env_smote_test))

print("\n\nRandom Forest Classifier (hyperparamteres) on training set with SMOTE (env)\n")
print(confusion_matrix(Y_env_train_smote, Y_pred_rf_h_env_smote_train))
print(classification_report(Y_env_train_smote, Y_pred_rf_h_env_smote_train))

#classe 0 74% classe 1 32%
print("\n\nRandom Forest Classifier (hyperparamteres) on test set with SMOTE (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_h_env_smote_test))
print(classification_report(Y_env_test, Y_pred_rf_h_env_smote_test))


##########################################################################
##################  FATTORI AMBIENTALI - ALTRI MODELLI  ##################
##########################################################################

#provo altri tipi di classificatori
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


svc_def_env_smote = SVC(random_state=42)

svc_def_env_smote=svc_def_env_smote.fit(X_env_train_smote, Y_env_train_smote)


Y_pred_svc_def_smote_train = svc_def_env_smote.predict(X_env_train_smote)
Y_pred_svc_def_smote_test = svc_def_env_smote.predict(X_env_test)

print("\n\nSVC (default hyperparamteres) on training set with SMOTE\n")
print(confusion_matrix(Y_train_smote, Y_pred_svc_def_smote_train))
print(classification_report(Y_train_smote, Y_pred_svc_def_smote_train))

#classe 0 59% classe 1 43% 
print("\n\nSVC (default hyperparamteres) on test set with SMOTE\n")
print(confusion_matrix(Y_test, Y_pred_svc_def_smote_test))
print(classification_report(Y_test, Y_pred_svc_def_smote_test))




gnm_def_env_smote = GaussianNB()
gnm_def_env_smote=gnm_def_env_smote.fit(X_env_train_smote, Y_env_train_smote)

Y_pred_gnm_def_smote_train = gnm_def_env_smote.predict(X_env_train_smote)
Y_pred_gnm_def_smote_test = gnm_def_env_smote.predict(X_env_test)

print("\n\nGaussianNB (default hyperparamteres) on training set with SMOTE\n")
print(confusion_matrix(Y_train_smote, Y_pred_gnm_def_smote_train))
print(classification_report(Y_train_smote, Y_pred_gnm_def_smote_train))

#classe 0 50% classe 1 60%
print("\n\nGaussianNB (default hyperparamteres) on test set with SMOTE\n")
print(confusion_matrix(Y_test, Y_pred_gnm_def_smote_test))
print(classification_report(Y_test, Y_pred_gnm_def_smote_test))

kn_def_env_smote = KNeighborsClassifier()
kn_def_env_smote=kn_def_env_smote.fit(X_env_train_smote, Y_env_train_smote)

Y_pred_kn_def_smote_train = kn_def_env_smote.predict(X_env_train_smote)
Y_pred_kn_def_smote_test = kn_def_env_smote.predict(X_env_test)

print("\n\nKNeighborsClassifier (default hyperparamteres) on training set with SMOTE\n")
print(confusion_matrix(Y_train_smote, Y_pred_kn_def_smote_train))
print(classification_report(Y_train_smote, Y_pred_kn_def_smote_train))

#classe 0 62% classe 1 46%
print("\n\nKNeighborsClassifier (default hyperparamteres) on test set with SMOTE\n")
print(confusion_matrix(Y_test, Y_pred_kn_def_smote_test))
print(classification_report(Y_test, Y_pred_kn_def_smote_test))

#print("\n\nricerca parametri knn\n\n")
#ricerca della combinazione migliore di parametri per il knn
#param_grid = {
#    'n_neighbors': [3, 5, 7, 9],
#    'weights': ['uniform', 'distance'],
#    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#    'leaf_size': [10, 20, 30]
#}
#grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
#grid.fit(X_env_train_smote, Y_env_train_smote)
#print("Best parameters found:", grid.best_params_)
#mi dice algorithm='auto', n_neighbors=3, weights='distance', leaf_size=10

kn_hyper_env_smote = KNeighborsClassifier(algorithm='auto', n_neighbors=3, weights='distance', leaf_size=10)
kn_hyper_env_smote=kn_hyper_env_smote.fit(X_env_train_smote, Y_env_train_smote)

Y_pred_kn_hyper_smote_train = kn_hyper_env_smote.predict(X_env_train_smote)
Y_pred_kn_hyper_smote_test = kn_hyper_env_smote.predict(X_env_test)

print("\n\nKNeighborsClassifier (hyperparamteres) on training set with SMOTE\n")
print(confusion_matrix(Y_env_train_smote, Y_pred_kn_hyper_smote_train))
print(classification_report(Y_env_train_smote, Y_pred_kn_hyper_smote_train))

#classe 0 65% classe 1 45%
print("\n\nKNeighborsClassifier (hyperparamteres) on test set with SMOTE\n")
print(confusion_matrix(Y_env_test, Y_pred_kn_hyper_smote_test))
print(classification_report(Y_env_test, Y_pred_kn_hyper_smote_test))


#provo a fare un undersampling della classe 0

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(random_state=42)
X_env_train_under, Y_env_train_under = undersample.fit_resample(X_env_train, Y_env_train)
print("\nY training set:\n")
print(Y_env_train.value_counts())
print("\nAfter applying undersampling:\n")
print(Y_env_train_under.value_counts())

rf_def_env_under = RandomForestClassifier(random_state=42)

rf_def_env_under.fit(X_env_train_under, Y_env_train_under)
Y_pred_rf_def_env_under_train = rf_def_env_under.predict(X_env_train)
Y_pred_rf_def_env_under_test = rf_def_env_under.predict(X_env_test)

print("\n\nRandom Forest Classifier (default hyperparamteres) on training set with undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_rf_def_env_under_train))
print(classification_report(Y_env_train, Y_pred_rf_def_env_under_train))

#classe 0 61% classe 1 42%
print("\n\nRandom Forest Classifier (default hyperparamteres) on test set with undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_def_env_under_test))
print(classification_report(Y_env_test, Y_pred_rf_def_env_under_test))

#ora provo a fare sia uno smote che un undersampling

smote_unbalanced = SMOTE(sampling_strategy=0.7, random_state=42)
X_env_train_smote_under, Y_env_train_smote_under = smote_unbalanced.fit_resample(X_env_train, Y_env_train)
X_env_train_smote_under, Y_env_train_smote_under = undersample.fit_resample(X_env_train_smote_under, Y_env_train_smote_under)

print("\nY training set:\n")
print(Y_env_train.value_counts())
print("\nAfter applying SMOTE and undersampling:\n")
print(Y_env_train_smote_under.value_counts())

rf_def_env_smote_under = RandomForestClassifier(random_state=42)
rf_def_env_smote_under.fit(X_env_train_smote_under, Y_env_train_smote_under)
Y_pred_rf_def_env_smote_under_train = rf_def_env_smote_under.predict(X_env_train)
Y_pred_rf_def_env_smote_under_test = rf_def_env_smote_under.predict(X_env_test)

print("\n\nRandom Forest Classifier (default hyperparamteres) on training set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_rf_def_env_smote_under_train))
print(classification_report(Y_env_train, Y_pred_rf_def_env_smote_under_train))

#classe 0 67% classe 1 40%
print("\n\nRandom Forest Classifier (default hyperparamteres) on test set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_def_env_smote_under_test))
print(classification_report(Y_env_test, Y_pred_rf_def_env_smote_under_test))

#questo è promettente
#vedo i parametri

#print("ricerca della combinazione migliore di parametri per il random forest")
#param_grid = {
#    'n_estimators': [50, 100, 200],
#    'max_depth': [3, 5, 10, None],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4]
#}
#gs_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
#gs_rf.fit(X_env_train_smote, Y_env_train_smote)
#print('Best parameters found by GridSearchCV:', gs_rf.best_params_)
#mi da n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1

rf_hyper_env_smote_under = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1, random_state=42)
rf_hyper_env_smote_under.fit(X_env_train_smote_under, Y_env_train_smote_under)

Y_pred_rf_hyper_env_smote_under_train = rf_hyper_env_smote_under.predict(X_env_train)
Y_pred_rf_hyper_env_smote_under_test = rf_hyper_env_smote_under.predict(X_env_test)

print("\n\nRandom Forest Classifier (hyperparamteres) on training set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_rf_hyper_env_smote_under_train))
print(classification_report(Y_env_train, Y_pred_rf_hyper_env_smote_under_train))

#classe 0 67% classe 1 39%
print("\n\nRandom Forest Classifier (hyperparamteres) on test set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_hyper_env_smote_under_test))
print(classification_report(Y_env_test, Y_pred_rf_hyper_env_smote_under_test))
#praticamente uguale a prima

#boh provo a fare gli altri modelli con lo smote e l'undersampling

svc_def_env_smote_under = SVC(random_state=42)
svc_def_env_smote_under=svc_def_env_smote_under.fit(X_env_train_smote_under, Y_env_train_smote_under)
Y_pred_svc_def_smote_under_train = svc_def_env_smote_under.predict(X_env_train)
Y_pred_svc_def_smote_under_test = svc_def_env_smote_under.predict(X_env_test)

print("\n\nSVC (default hyperparamteres) on training set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_svc_def_smote_under_train))
print(classification_report(Y_env_train, Y_pred_svc_def_smote_under_train))

# 0 61% 1 42%
print("\n\nSVC (default hyperparamteres) on test set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_svc_def_smote_under_test))
print(classification_report(Y_env_test, Y_pred_svc_def_smote_under_test))

gnm_def_env_smote_under = GaussianNB()
gnm_def_env_smote_under=gnm_def_env_smote_under.fit(X_env_train_smote_under, Y_env_train_smote_under)
Y_pred_gnm_def_smote_under_train = gnm_def_env_smote_under.predict(X_env_train)
Y_pred_gnm_def_smote_under_test = gnm_def_env_smote_under.predict(X_env_test)

print("\n\nGaussianNB (default hyperparamteres) on training set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_gnm_def_smote_under_train))
print(classification_report(Y_env_train, Y_pred_gnm_def_smote_under_train))

# 0 49% 1 45%
print("\n\nGaussianNB (default hyperparamteres) on test set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_gnm_def_smote_under_test))
print(classification_report(Y_env_test, Y_pred_gnm_def_smote_under_test))

kn_def_env_smote_under = KNeighborsClassifier()
kn_def_env_smote_under=kn_def_env_smote_under.fit(X_env_train_smote_under, Y_env_train_smote_under)
Y_pred_kn_def_smote_under_train = kn_def_env_smote_under.predict(X_env_train)
Y_pred_kn_def_smote_under_test = kn_def_env_smote_under.predict(X_env_test)

print("\n\nKNeighborsClassifier (default hyperparamteres) on training set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_kn_def_smote_under_train))
print(classification_report(Y_env_train, Y_pred_kn_def_smote_under_train))

#0 61% 1 45%
print("\n\nKNeighborsClassifier (default hyperparamteres) on test set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_kn_def_smote_under_test))
print(classification_report(Y_env_test, Y_pred_kn_def_smote_under_test))

#il knn con smote e undersampling sembra essere il migliore
#provo a fare la ricerca dei parametri

#print("\n\nricerca parametri knn\n\n")
#param_grid = {
#    'n_neighbors': [6, 7, 8],
#    'weights': ['uniform', 'distance'],
#    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
#    'leaf_size': [1, 2, 5]
#}
#grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
#grid.fit(X_env_train_smote_under, Y_env_train_smote_under)
#print("Best parameters found:", grid.best_params_)
#mi da n_neighbors=7, weights='distance', algorithm='ball_tree', leaf_size=3
#in realtà il leaf size me lo dava a 1 ma mi puzza che sia così piccolo

kn_hyper_env_smote_under = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='ball_tree', leaf_size=3)
kn_hyper_env_smote_under=kn_hyper_env_smote_under.fit(X_env_train_smote_under, Y_env_train_smote_under)
Y_pred_kn_hyper_smote_under_train = kn_hyper_env_smote_under.predict(X_env_train)
Y_pred_kn_hyper_smote_under_test = kn_hyper_env_smote_under.predict(X_env_test)

print("\n\nKNeighborsClassifier (hyperparamteres) on training set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_kn_hyper_smote_under_train))
print(classification_report(Y_env_train, Y_pred_kn_hyper_smote_under_train))

# 0 61% 1 47%
print("\n\nKNeighborsClassifier (hyperparamteres) on test set with SMOTE and undersampling (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_kn_hyper_smote_under_test))
print(classification_report(Y_env_test, Y_pred_kn_hyper_smote_under_test))

#non so come proseguire


