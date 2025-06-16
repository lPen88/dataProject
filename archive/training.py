#stesse identiche cose che ho fatto nell'altro progetto

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('dataset/mainDataset.csv')

#ci stanno alcune entry con valori nulli, le droppo
df = df.dropna()

#estrae la colonna con la categoria di rischio
Y = df['Heart Attack Risk']

#la rimuovo dal resto del dataset
X = df.drop(columns=['Heart Attack Risk'])

#prima di addestrare il modello devo convertire la colonna gender in valori numerici
#1 per maschio e 0 per femmina
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#creo il modello di regressione lineare
lr = LinearRegression()
lr.fit(X_train, Y_train)

#facciamo una previsione sul dataset originale
Y_train_pred = lr.predict(X_train)

#confrontiamo le previsioni con i valori reali
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_train, Y_train_pred)
r2 = r2_score(Y_train, Y_train_pred)
print(f'linear Mean Squared Error: {mse}')
print(f'linear R^2 Score: {r2}')


#poi di nuovo faccio un modello di classificazione
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#addestro il modello
rf = GradientBoostingClassifier(random_state=42)
rf.fit(X_train, Y_train)

#lo testo sullo stesso dataset di addestramento, che mi immagino debba uscire perfetto o quasi
Y_train_pred = rf.predict(X_train)

#lo valuto
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_train, Y_train_pred))
print(classification_report(Y_train, Y_train_pred))

#come detto in questo commento https://www.kaggle.com/code/alikalwar/heart-attack-risk-prediction/comments#3125675
#il modello ha preso 0 come uscita generale per tutte le entry, infatti il recall di 0 è 100% mentre di 1 è 6%

#provo la soluzione suggerita nel commento

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
rf.fit(X_train_resampled, Y_train_resampled)


Y_train_pred = rf.predict(X_train)

print(confusion_matrix(Y_train, Y_train_pred))
print(classification_report(Y_train, Y_train_pred))

#il recall è migliorato ma comunque la predizione non è perfetta

# invece di usare SMOTE faccio l'undersampling per la classe 0

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, Y_train_resampled = rus.fit_resample(X_train, Y_train)

# essenzialmente così il dataset di addestramento ha lo stesso numero di entry per entrambe le classi

print(X_train_resampled.value_counts())
print(Y_train_resampled.value_counts())

rf.fit(X_train_resampled, Y_train_resampled)


Y_train_pred = rf.predict(X_train)

print(confusion_matrix(Y_train, Y_train_pred))
print(classification_report(Y_train, Y_train_pred))

# la precision per la classe 1 è salita al 50%, boh non so come altro migliorare
# ora faccio la previsione sul test set

print("\n\nTest Set Prediction\n")
Y_test_pred = rf.predict(X_test)
print(confusion_matrix(Y_test, Y_test_pred))
print(classification_report(Y_test, Y_test_pred))

#che è alquanto pessimo
#provo a cambiare il modello di classificazione
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, Y_train_resampled)

rf.fit(X_train_resampled, Y_train_resampled)


Y_train_pred = rf.predict(X_train)

print(confusion_matrix(Y_train, Y_train_pred))
print(classification_report(Y_train, Y_train_pred))

#che ha una precisione ottima per entrambe le classi, ma mi puzza di overfitting
#vedo sul test
Y_test_pred = rf.predict(X_test)
print(confusion_matrix(Y_test, Y_test_pred))
print(classification_report(Y_test, Y_test_pred))
#infatti sul test la precisione scende al 70% per classe 0 e 40% per 1