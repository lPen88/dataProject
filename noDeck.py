import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV

# prova senza i campi relativi al ponte

df = pd.read_csv('dataset/train_processed.csv')

df=df.drop(columns=['Deck_A', 'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Deck_X'])

X= df.drop(columns=['Survived'])
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression



search_space = {
    'C': (1e-3, 100.0, 'log-uniform'),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

opt = BayesSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    search_spaces=search_space,
    n_iter=32,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

opt.fit(X_train, Y_train)

print("Best Parameters Found:", opt.best_params_)
print("\nClassification Report on Test Set:")
print(classification_report(Y_test, opt.predict(X_test)))

test_df = pd.read_csv('dataset/test_processed.csv')
test_df = test_df.drop(columns=['Deck_A', 'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Deck_X'])
passegers_ids = pd.read_csv('dataset/test.csv')['PassengerId']

Y_test_final = opt.predict(test_df)


submission_df = pd.DataFrame({
    'PassengerId': passegers_ids,
    'Survived': Y_test_final
})

submission_df['Survived'] = submission_df['Survived'].astype(int)
submission_df.to_csv('del.csv', index=False)

# 0.784 (!!!)