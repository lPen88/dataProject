import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Caricamento dei dati
df = pd.read_csv('dataset/mainDataset2.csv')

# Preprocessing
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# Selezione delle feature cliniche
clinical_features = [
    'Cholesterol', 'Triglycerides', 'Blood sugar', 'CK-MB', 'Troponin',
    'Systolic blood pressure', 'Diastolic blood pressure', 'Heart rate', 'BMI'
]
X = df[clinical_features]
Y = df['Heart Attack Risk']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Bilanciamento: SMOTE + undersampling
smote = SMOTE(random_state=42, sampling_strategy=0.85)
undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.9)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
X_balanced, Y_balanced = undersampler.fit_resample(X_train_smote, Y_train_smote)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_balanced)
X_test_scaled = scaler.transform(X_test)

# Addestramento Random Forest pesato
rf_weighted_best = RandomForestClassifier(
    class_weight={0: 1, 1: 3},
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf_weighted_best.fit(X_train_scaled, Y_balanced)

# Predizione e valutazione
y_pred_rf = rf_weighted_best.predict(X_test_scaled)

print("Random Forest Ottimizzata:")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(Y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(Y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(Y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(Y_test, y_pred_rf):.4f}")
print("\nClassification Report:\n", classification_report(Y_test, y_pred_rf))
