import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset/train_processed.csv')

X= df.drop(columns=['Survived'])
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


from sklearn.metrics import classification_report, confusion_matrix


# provando vari modelli
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

import matplotlib.pyplot as plt

for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))
    print("Classification Report:")
    print(classification_report(Y_test, y_pred))

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, Y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f"Learning Curve: {name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# hyperparameter tuning for Random Forest

from skopt import BayesSearchCV

search_space = {
    'n_estimators': (50, 300),
    'max_depth': (3, 30),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

# Ottimizzazione bayesiana
opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=32,
    cv=3,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

# Addestramento
opt.fit(X_train, Y_train)

# Valutazione
print("Best Parameters Found:", opt.best_params_)
print("\nClassification Report on Test Set:")
print(classification_report(Y_test, opt.predict(X_test)))


# test sul dataset di test
test_df = pd.read_csv('dataset/test_processed.csv')
passegers_ids = pd.read_csv('dataset/test.csv')['PassengerId']

Y_test_final = opt.predict(test_df)


submission_df = pd.DataFrame({
    'PassengerId': passegers_ids,
    'Survived': Y_test_final
})

# il collega li vuole in int
submission_df['Survived'] = submission_df['Survived'].astype(int)
submission_df.to_csv('dataset/result/submission.csv', index=False)



# stesso con il dataset alternativo (ha il titolo)

test_df_alt = pd.read_csv('dataset/train_processed_alt.csv')

X = test_df_alt.drop(columns=['Survived'])
Y = test_df_alt['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=32,
    cv=3,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

opt.fit(X_train, Y_train)

print("Best Parameters Found (with title):", opt.best_params_)
print("\nClassification Report on Test Set:")
print(classification_report(Y_test, opt.predict(X_test)))

test_df = pd.read_csv('dataset/test_processed_alt.csv')
passegers_ids = pd.read_csv('dataset/test.csv')['PassengerId']

Y_test_final = opt.predict(test_df)


submission_df = pd.DataFrame({
    'PassengerId': passegers_ids,
    'Survived': Y_test_final
})

# il collega li vuole in int
submission_df['Survived'] = submission_df['Survived'].astype(int)
submission_df.to_csv('dataset/result/submission_alt.csv', index=False)