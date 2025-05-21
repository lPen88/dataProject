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
#gs_rf.fit(X_train, Y_train)
#
#print('Best parameters found by GridSearchCV:', gs_rf.best_params_)
from sklearn.metrics import classification_report, confusion_matrix

rf_hyper = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_hyper.fit(X_train, Y_train)

rf_def = RandomForestClassifier(random_state=42)
rf_def.fit(X_train, Y_train)

#Y prediction random forest (default hyperparameters) on training set
Y_pred_rf_def_train = rf_def.predict(X_train)
Y_pred_rf_def_test = rf_def.predict(X_test)

#Y prediction random forest (with hyperparameters) on training set
Y_pred_rf_h_train = rf_hyper.predict(X_train)
Y_pred_rf_test = rf_hyper.predict(X_test)

print("Random Forest Classifier (default hyperparamteres) on training set\n")
print(confusion_matrix(Y_train, Y_pred_rf_def_train))
print(classification_report(Y_train, Y_pred_rf_def_train))

print("\n\nRandom Forest Classifier (default hyperparamteres) on test set\n")
print(confusion_matrix(Y_test, Y_pred_rf_def_test))
print(classification_report(Y_test, Y_pred_rf_def_test))


print("\n\nRandom Forest Classifier (hyperparamteres) on training set\n")
print(confusion_matrix(Y_train, Y_pred_rf_h_train))
print(classification_report(Y_train, Y_pred_rf_h_train))

print("\n\nRandom Forest Classifier (hyperparamteres) on test set\n")
print(confusion_matrix(Y_test, Y_pred_rf_test))
print(classification_report(Y_test, Y_pred_rf_test))

print("\nentrambi i modelli sono overfittati sul training set")

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.85)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
print("\nY training set:\n")
print(Y_train.value_counts())
print("\nAfter applying SMOTE:\n")
print(Y_train_smote.value_counts())

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

print("modello basato su fattori ambientali")

X_env=df[['Age', 'Gender', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week' , 'Previous Heart Problems' , 'Medication Use', 'Stress Level' , 'Sedentary Hours Per Day', 'BMI', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
Y_env = df['Heart Attack Risk']
X_env['Gender'] = X_env['Gender'].map({'Male': 1, 'Female': 0})

X_env_train, X_env_test, Y_env_train, Y_env_test = train_test_split(X_env, Y_env, test_size=0.2, random_state=42)

rf_def_env = RandomForestClassifier(random_state=42)
rf_hyper_env = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=42)

rf_def_env.fit(X_env_train, Y_env_train)
rf_hyper_env.fit(X_env_train, Y_env_train)

Y_pred_rf_def_env_train = rf_def_env.predict(X_env_train)
Y_pred_rf_def_env_test = rf_def_env.predict(X_env_test)

Y_pred_rf_h_env_train = rf_hyper_env.predict(X_env_train)
Y_pred_rf_h_env_test = rf_hyper_env.predict(X_env_test)

print("\n\nRandom Forest Classifier (default hyperparamteres) on training (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_rf_def_env_train))
print(classification_report(Y_env_train, Y_pred_rf_def_env_train))

print("\n\nRandom Forest Classifier (default hyperparamteres) on test (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_def_env_test))
print(classification_report(Y_env_test, Y_pred_rf_def_env_test))

print("\n\nRandom Forest Classifier (hyperparamteres) on training (env)\n")
print(confusion_matrix(Y_env_train, Y_pred_rf_h_env_train))
print(classification_report(Y_env_train, Y_pred_rf_h_env_train))

print("\n\nRandom Forest Classifier (hyperparamteres) on test (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_h_env_test))
print(classification_report(Y_env_test, Y_pred_rf_h_env_test))

#provo anche qui ad usare smote
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
#
#print('Best parameters found by GridSearchCV:', gs_rf.best_params_)

#qui mi ha dato invece
#n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1

rf_def_env_smote = RandomForestClassifier(random_state=42)
rf_hyper_env_smote = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
rf_def_env_smote.fit(X_env_train_smote, Y_env_train_smote)
rf_hyper_env_smote.fit(X_env_train_smote, Y_env_train_smote)

Y_pred_rf_def_env_smote_train = rf_def_env_smote.predict(X_env_train_smote)
Y_pred_rf_def_env_smote_test = rf_def_env_smote.predict(X_env_test)

Y_pred_rf_h_env_smote_train = rf_hyper_env_smote.predict(X_env_train_smote)
Y_pred_rf_h_env_smote_test = rf_hyper_env_smote.predict(X_env_test)

print("\n\nRandom Forest Classifier (default hyperparamteres) on training set with SMOTE (env)\n")
print(confusion_matrix(Y_env_train_smote, Y_pred_rf_def_env_smote_train))
print(classification_report(Y_env_train_smote, Y_pred_rf_def_env_smote_train))

print("\n\nRandom Forest Classifier (default hyperparamteres) on test set with SMOTE (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_def_env_smote_test))
print(classification_report(Y_env_test, Y_pred_rf_def_env_smote_test))

print("\n\nRandom Forest Classifier (hyperparamteres) on training set with SMOTE (env)\n")
print(confusion_matrix(Y_env_train_smote, Y_pred_rf_h_env_smote_train))
print(classification_report(Y_env_train_smote, Y_pred_rf_h_env_smote_train))

print("\n\nRandom Forest Classifier (hyperparamteres) on test set with SMOTE (env)\n")
print(confusion_matrix(Y_env_test, Y_pred_rf_h_env_smote_test))
print(classification_report(Y_env_test, Y_pred_rf_h_env_smote_test))

#lo score è salito al 30% 🔥🔥🔥
