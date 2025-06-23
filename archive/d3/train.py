import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier


# Caricamento del dataset
df = pd.read_csv("Heart_Attack_Prediction.csv")

# Mappatura della colonna 'Gender'
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Encoding di eventuali altre colonne categoriche
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Heart_Attack_Risk':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

print(df['Heart_Attack_Risk'].value_counts())

# Definizione dei cluster di feature
lifestyle_features = ['Age', 'Gender', 'Smoking', 'Alcohol_Consumption', 'Physical_Activity', 'Diet_Score', 'Stress_Level', 'Obesity', 'Family_History', 'Heart_Attack_History']
clinical_features = ['Age', 'Gender', 'Cholesterol_Level', 'LDL_Level', 'HDL_Level', 'Systolic_BP', 'Diastolic_BP']

X = df[lifestyle_features]
Y = df['Heart_Attack_Risk']

import matplotlib.pyplot as plt

#dataset = X.copy()
#dataset['Heart Attack Risk'] = Y
#
#risk_counts_both = {}
#risk_counts_obese = {}
#risk_counts_activity = {}
#risk_counts_neither = {}
#for risk_class in [0, 1]:
#    count_both = dataset[
#        (dataset['Obesity'] == 1) &
#        (dataset['Physical_Activity'] == 1) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_obese = dataset[
#        (dataset['Obesity'] == 1) &
#        (dataset['Physical_Activity'] == 0) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_activity = dataset[
#        (dataset['Obesity'] == 0) &
#        (dataset['Physical_Activity'] == 1) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_neither = dataset[
#        (dataset['Obesity'] == 0) &
#        (dataset['Physical_Activity'] == 0) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    risk_counts_both[risk_class] = count_both
#    risk_counts_obese[risk_class] = count_obese
#    risk_counts_activity[risk_class] = count_activity
#    risk_counts_neither[risk_class] = count_neither
#
#class_0_total = dataset[dataset['Heart Attack Risk'] == 0].shape[0]
#class_1_total = dataset[dataset['Heart Attack Risk'] == 1].shape[0]
#
## Prepare data for plotting
#labels = ['Both', 'Obese', 'Physically Active', 'Neither']
#counts_0 = [risk_counts_both[0]*100/class_0_total, 
#            risk_counts_obese[0]*100/class_0_total, 
#            risk_counts_activity[0]*100/class_0_total, 
#            risk_counts_neither[0]*100/class_0_total]
#counts_1 = [risk_counts_both[1]*100/class_1_total, 
#            risk_counts_obese[1]*100/class_1_total, 
#            risk_counts_activity[1]*100/class_1_total, 
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
#ax.set_title('Percentage of Heart Attack Risk by obesity and physical activity')
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

#gender_counts = X['Gender'].value_counts()
#plt.bar(gender_counts.index.astype(str), gender_counts.values, color=['pink', 'skyblue'])
#plt.xticks([0, 1], ['Female', 'Male'])
#plt.xlabel('Gender')
#plt.ylabel('Count')
#plt.title('Count of Gender (0=Female, 1=Male)')
#plt.show()
#
#plt.hist(X['Age'], bins=20, color='skyblue', edgecolor='black')
#plt.xlabel('Age')
#plt.ylabel('Count')
#plt.title('Distribution of Age')
#plt.show()
#
#dataset = X.copy()
#dataset['Heart Attack Risk'] = Y
#
#risk_counts_both = {}
#risk_counts_smoking = {}
#risk_counts_alcohol = {}
#risk_counts_neither = {}
#for risk_class in [0, 1]:
#    count_both = dataset[
#        (dataset['Smoking'] == 1) &
#        (dataset['Alcohol_Consumption'] == 1) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_smoking = dataset[
#        (dataset['Smoking'] == 1) &
#        (dataset['Alcohol_Consumption'] == 0) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_alcohol = dataset[
#        (dataset['Smoking'] == 0) &
#        (dataset['Alcohol_Consumption'] == 1) &
#        (dataset['Heart Attack Risk'] == risk_class)
#    ].shape[0]
#    count_neither = dataset[
#        (dataset['Smoking'] == 0) &
#        (dataset['Alcohol_Consumption'] == 0) &
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


#X=X.drop(columns=['Smoking', 'Alcohol_Consumption', 'Obesity', 'Physical_Activity'])

smote = SMOTE(random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)



xgb = XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_smote, Y_train_smote)

print("XGBoost Classifier:")
print("Train Set:")
print(confusion_matrix(Y_train_smote, xgb.predict(X_train_smote)))
print(classification_report(Y_train_smote, xgb.predict(X_train_smote)))
print("Test Set:")
print(confusion_matrix(Y_test, xgb.predict(X_test)))
print(classification_report(Y_test, xgb.predict(X_test)))

rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=1, min_samples_split=4,random_state=42)
rf.fit(X_train_smote, Y_train_smote)

print("Random Forest Classifier:")
print("Train Set:")
print(confusion_matrix(Y_train_smote, rf.predict(X_train_smote)))
print(classification_report(Y_train_smote, rf.predict(X_train_smote)))
print("Test Set:")
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

# Predict probabilities for the test set
y_proba = rf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(Y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()

