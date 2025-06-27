import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

df = pd.concat([train_df, test_df], ignore_index=True)


#gestione mancanti, mancano solo per due passeggeri, cercandole su internet
#Martha Evelyn Stone (830) si è imbarcata a Southampton assieme la domestica Amelie Icard (62)
df.loc[df['PassengerId'] == 830, 'Embarked'] = 'S'
df.loc[df['PassengerId'] == 62, 'Embarked'] = 'S'

#gestione "Fare" mancante, una sola entry è assente
#Thomas Storey (1044) in 3° classe, imbarcato senza parenti
#prendo il valore mediano della tariffa per 3° classe di tutti i passeggeri con SibSp=0 e Parch=0
df.loc[df['PassengerId'] == 1044, 'Fare'] = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]


# gestione età mancanti, utilizzo della mediana per sesso e classe
median_ages = df.groupby(['Sex', 'Pclass'])['Age'].median()
print(median_ages)
df['Age'] = df.apply(
    lambda row: median_ages[row['Sex'], row['Pclass']] if pd.isna(row['Age']) else row['Age'],
    axis=1
)

# gestione "Cabin" mancante, questa è più complessa
# secondo tutto internet*, la lettera della cabina indica il ponte, quindi invece di recuperare
# la cabina esatta mi segno su che ponte si trovava il passeggero
# inoltre, questo https://www.encyclopedia-titanica.org/titanic-deckplans dice che
# i ponti A, B e C erano occupate solo da cabine di prima classe,
# D e E da cabine di tutte le classi
# F e G occupate da cabine di seconda e terza classe

# poi sul ponte più alto (nell'immagine chiamato "Boat Deck") ci sono le cabine T, U, W, X, Y, Z
# nel dataset compare una sola volta la cabina T di uno listato in prima classe, sopravvissuto
# per evitare di avere un ponte con una singola entry, lo considero come ponte A


# *trovare una fonte primara per questo mi è stato alquanto difficile, questa rivista del 1911 non lo dice direttamente ma lo implica
# https://www.ggarchives.com/OceanTravel/Titanic/01-PlanningBuildingLaunching/Decks-ComprehensiveDetails.html

df['Deck'] = df['Cabin'].apply(lambda x: 'A' if pd.notna(x) and x[0] == 'T' else (x[0] if pd.notna(x) else 'X'))

# le cabine non le uso più
df=df.drop(columns=['Cabin'])

# poi per SibSp e ParCh li sommo per vedere quanta gente sta in una famiglia

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 per includere il passeggero stesso

df=df.drop(columns=['SibSp', 'Parch'])

print(df.info())

# il nome non avevo intenzione di usarlo ma poi ho pensato di vedere se il titolo del passeggero migliora le performance
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)


# encoding delle feature
le = LabelEncoder()

df=df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked'])

df = pd.get_dummies(df, columns=['Pclass', 'Deck'], prefix=['Pclass', 'Deck'])
df.loc[:, 'Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Fare'] = pd.qcut(df['Fare'], 10)
df['Age'] = pd.qcut(df['Age'], 10)
df['Fare'] = le.fit_transform(df['Fare'].astype(str))
df['Age'] = le.fit_transform(df['Age'].astype(str))

# Bin FamilySize in 4 classi
def bin_family_size(size):
    if size == 1:
        return 0  # solo
    elif 2 <= size <= 4:
        return 1  # piccola
    elif 5 <= size <= 8:
        return 2  # media
    else:
        return 3  # grande

df.loc[:, 'FamilySize'] = df['FamilySize'].apply(bin_family_size)
df = pd.get_dummies(df, columns=['FamilySize'], prefix='FamilySize')

# questo dataset a parte ha il titolo
df_with_title = df.copy()
df=df.drop(columns=['Title'])

# li ragruppo in base alla tipologia, scelte in base al tasso di sopravvivenza (guarda in test)
df_with_title['Title'] = df_with_title['Title'].replace(
    {
        'Mrs': 'Miss',
        'Ms': 'Miss',
        'Mlle': 'Miss',
        'Mme': 'Miss',
        'Master': 'Mr',
        'Dona': 'Nobility',
        'Don': 'Nobility',
        'Jonkheer': 'Nobility',
        'the Countess': 'Nobility',
        'Lady': 'Nobility',
        'Sir': 'Nobility',
        'Rev': 'Clergy',
        'Col': 'Special',
        'Major': 'Special',
        'Capt': 'Special',
        'Dr': 'Special'
    }
)

df_with_title = pd.get_dummies(df_with_title, columns=['Title'], prefix='Title')

print(df.head(1))
print(df_with_title.head(1))


df_test = df[df['Survived'].isnull()].copy()
df_test = df_test.drop(columns=['Survived'])
df_train = df[df['Survived'].notnull()].copy()

df_with_title_test = df_with_title[df_with_title['Survived'].isnull()].copy()
df_with_title_test = df_with_title_test.drop(columns=['Survived'])
df_with_title_train = df_with_title[df_with_title['Survived'].notnull()].copy()

df_train.to_csv('dataset/train_processed.csv', index=False)
df_test.to_csv('dataset/test_processed.csv', index=False)

df_with_title_train.to_csv('dataset/train_processed_alt.csv', index=False)
df_with_title_test.to_csv('dataset/test_processed_alt.csv', index=False)