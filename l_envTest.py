#modello basato sui fattori ambientali
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('dataset/dataset_l_fix.csv')

df = df[df['Sleep Hours Per Day'] != 0]
df.loc[:, 'Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df.loc[:, 'Diet'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})

le = LabelEncoder()

Y=df['Heart Attack Risk']
X=df.drop(columns=['Country', 'Continent', 'Hemisphere', 'Heart Attack Risk'])
X=X[['Cholesterol', 'Exercise Hours Per Week', 'Sedentary Hours Per Day',
       'Income', 'BMI', 'Triglycerides']]

#X.loc[:, 'Diet'] = le.fit_transform(X['Diet'])

#normalizing columns 'Cholesterol' and 'BMI'
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X.loc[:, 'Triglycerides'] = scaler.fit_transform(X[['Triglycerides']])
X.loc[:, 'Cholesterol'] = scaler.fit_transform(X[['Cholesterol']])
X.loc[:, 'BMI'] = scaler.fit_transform(X[['BMI']])

print(X.head())

import numpy as np
from matplotlib import pyplot as plt

dataset = X.copy()
dataset['Heart Attack Risk'] = Y

#risk_counts_both = {}
#risk_counts_smoking = {}
#risk_counts_alcohol = {}
#risk_counts_neither = {}
#for risk_class in [0, 1]:
#    count_both = dataset[
#        (dataset['Smoking'] == 1) &
#        (dataset['Alcohol Consumption'] == 1) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_smoking = dataset[
#        (dataset['Smoking'] == 1) &
#        (dataset['Alcohol Consumption'] == 0) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_alcohol = dataset[
#        (dataset['Smoking'] == 0) &
#        (dataset['Alcohol Consumption'] == 1) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_neither = dataset[
#        (dataset['Smoking'] == 0) &
#        (dataset['Alcohol Consumption'] == 0) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    risk_counts_both[risk_class] = count_both
#    risk_counts_smoking[risk_class] = count_smoking
#    risk_counts_alcohol[risk_class] = count_alcohol
#    risk_counts_neither[risk_class] = count_neither
#
#class_0_total = dataset[dataset['Heart Attack Risk'] == 0].shape[0]
#class_1_total = dataset[dataset['Heart Attack Risk'] == 1].shape[0]
#
## Prepare data for plotting
#labels = ['Both', 'Smoking only', 'Alcohol only', 'Neither']
#counts_0 = [risk_counts_both[0]*100/class_0_total, 
#            risk_counts_smoking[0]*100/class_0_total, 
#            risk_counts_alcohol[0]*100/class_0_total, 
#            risk_counts_neither[0]*100/class_0_total]
#counts_1 = [risk_counts_both[1]*100/class_1_total, 
#            risk_counts_smoking[1]*100/class_1_total, 
#            risk_counts_alcohol[1]*100/class_1_total, 
#            risk_counts_neither[1]*100/class_1_total]
#
#x = np.arange(len(labels))  # label locations
#width = 0.35  # width of the bars
#
#fig, ax = plt.subplots(figsize=(8, 6))
#rects1 = ax.bar(x - width/2, counts_0, width, label='Risk 0', color='blue')
#rects2 = ax.bar(x + width/2, counts_1, width, label='Risk 1', color='red')
#
## Add labels, title, and custom x-axis tick labels
#ax.set_ylabel('Percentage (%)')
#ax.set_title('Percentage of Heart Attack Risk by Smoking and Alcohol Consumption')
#ax.set_xticks(x)
#ax.set_xticklabels(labels)
#ax.legend()
#
## Add value labels on bars with two decimal points
#for rects in [rects1, rects2]:
#    for rect in rects:
#        height = rect.get_height()
#        ax.annotate(f'{height:.2f}',
#                    xy=(rect.get_x() + rect.get_width() / 2, height),
#                    xytext=(0, 3),  # 3 points vertical offset
#                    textcoords="offset points",
#                    ha='center', va='bottom')
#
#plt.tight_layout()
#plt.show()

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

smote = SMOTE()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

#X_with_Y_b = X_balanced.copy()
#X_with_Y_b['Heart Attack Risk'] = Y_balanced
#
#X_with_Y_b.to_csv('dataset/X_with_Y_b.csv', index=False)

#print(Y_train.value_counts())
#print(Y_train_smote.value_counts())

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

rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, min_samples_leaf=2, random_state=42)


rf.fit(X_train_smote, Y_train_smote)

print('Random Forest')
print('train set')
print(confusion_matrix(Y_train_smote, rf.predict(X_train_smote)))
print(classification_report(Y_train_smote, rf.predict(X_train_smote)))
print('test set')
print(confusion_matrix(Y_test, rf.predict(X_test)))
print(classification_report(Y_test, rf.predict(X_test)))

#from skopt import BayesSearchCV
#
#param_space = {
#    'n_estimators': (50, 300),
#    'max_depth': (5, 50),     
#    'min_samples_split': (1, 20),
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

#from sklearn.model_selection import cross_val_score

##Initialize your model
#rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, min_samples_leaf=2)
#
## Perform 5-fold cross-validation
#scores = cross_val_score(rf, X, Y, cv=5)
#
#print("Cross-validation scores:", scores)
#print("Average score:", scores.mean())

#from sklearn.feature_selection import RFE
#
#rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=6)
#X_new = rfe.fit_transform(X, Y)
#selected_features = X.columns[rfe.support_]
#print(selected_features)

#import tensorflow as tf
#
## Define your model architecture
#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_smote.shape[1],)),
#    tf.keras.layers.Dense(8, activation='relu'),
#    tf.keras.layers.Dense(1, activation='sigmoid')  # Use 'softmax' and adjust units for multiclass
#])
#
## Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
## Train the model
#model.fit(X_train_smote, Y_train_smote, epochs=16, batch_size=8, validation_split=0.2)
#
#loss, accuracy = model.evaluate(X_test, Y_test)
#
#print(f"ANN Test Loss: {loss:.4f}")
#print(f"ANN Test Accuracy: {accuracy:.4f}")