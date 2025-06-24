import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

df = pd.concat([train_df, test_df], ignore_index=True)

#tutti i grafici li ho fatti per vedere cosa poteva essere rilevante per il training
#mo li lascio qui dentro così quando scriviamo il report basta copiarli

##distribuzione età in base a sesso e classe
#g = sns.FacetGrid(df, row="Sex", col="Pclass", margin_titles=True, height=3, aspect=1.2)
#g.map(sns.histplot, "Age", bins=40, color="steelblue", alpha=0.7, kde=False)
## per avere più tacche sull'asse x
#for ax in g.axes.flatten():
#    ax.set_xticks(np.arange(0, 85, 10))  # Adjust range and step as needed
#plt.subplots_adjust(top=0.9)
#g.figure.suptitle("Distribuzione dell'età per sesso e classe (tutti i passeggeri)")
#plt.show()


##distribuzione età in base a sesso, classe e sopravvivenza
#df_plot = df.dropna(subset=['Survived', 'Age'])
#
## Convert 'Survived' to string for better plot labels
#df_plot['Survived'] = df_plot['Survived'].map({0: 'Not Survived', 1: 'Survived'})
#
## Plot: Age distribution by Survived, Sex, and Pclass
#g = sns.FacetGrid(df_plot, row="Sex", col="Pclass", hue="Survived", margin_titles=True, height=3, aspect=1.2)
#g.map(sns.histplot, "Age", bins=40, alpha=0.7, kde=False)
#g.add_legend(title="Survival Status")
#plt.subplots_adjust(top=0.9)
#g.figure.suptitle("Distribuzione dell'età per sesso, classe e sopravvivenza")
#plt.show()

#gestione "embarked" mancanti, mancano solo per due passeggeri, cercandole su internet
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

# composizione dei ponti in base alla classe

#deck_pclass_counts = df.groupby(['Deck', 'Pclass']).size().unstack(fill_value=0)
#deck_totals = deck_pclass_counts.sum(axis=1)
#deck_pclass_percent = deck_pclass_counts.divide(deck_totals, axis=0) * 100
#pclass_colors = {1: '#d8a6a6', 2: '#c46666', 3: '#a00000'}
#colors = [pclass_colors[p] for p in sorted(deck_pclass_percent.columns)]
#ax = deck_pclass_percent.plot(kind='bar', stacked=True, color=colors)
#plt.ylabel('Deck composition (%)')
#plt.title('Percentage of Pclass per Deck')
#plt.tight_layout()
## il bastardo voleva le etichette verticali
#ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#plt.show()

# dal disegnetto vedi che quelle con cabin mancante sono per lo più di terza classe


# percentuale di sopravvissuti per ponte

#survival_rate_per_deck = df.groupby('Deck')['Survived'].mean() * 100
#survival_rate_per_deck = survival_rate_per_deck.sort_index()
#
#plt.figure(figsize=(8, 5))
#
#for i, deck in enumerate(survival_rate_per_deck.index):
#    plt.gca().add_patch(
#        plt.Rectangle((i - 0.4, 0), 0.8, 100, color='#c46666', alpha=0.15, zorder=0)
#    )
#sns.barplot(x=survival_rate_per_deck.index, y=survival_rate_per_deck.values, color="steelblue", zorder=1)
#plt.ylabel('Percentage of Survivors (%)')
#plt.xlabel('Deck')
#plt.title('Percentage of Survivors per Deck')
#plt.ylim(0, 100)
#plt.tight_layout()
#plt.show()

# IL GRAFICO SOPRA E' MISLEADING
# la percentuale di sopravvivenza per il ponte G, composto da cabine di seconda e terza classe, è più alto del ponte A
# ma se guardiamo il tasso di sopravvivenza per classe, vediamo che la terza classe ha un tasso di sopravvivenza molto basso
# ciò è dovuto al fatto che le entry per il ponte G sono letteralmente 5, di cui 2 sono sopravvissute


# visto che il ponte X è composto da tutte e tre le classi, vediamo la differenza di sopravvivenza per classe su quel ponte

#deck_x = df[df['Deck'] == 'X']
#survival_pclass = deck_x.groupby('Pclass')['Survived'].mean() * 100
#survival_pclass = survival_pclass.sort_index()
#
#plt.figure(figsize=(8, 5))
#sns.barplot(x=survival_pclass.index, y=survival_pclass.values, color="steelblue")
#plt.ylabel('Percentage of Survivors (%)')
#plt.xlabel('Pclass')
#plt.title('Percentage of Survivors per Pclass (Deck X)')
#plt.tight_layout()
#plt.show()

# come prevedibile, la sopravvivenza è più alta per la prima classe e seconda classe, visto che con molta probabilità
# le cabine, nonostante ignote, sappiamo che si troveranno tra il ponte A e E, che hanno un tasso di sopravvivenza più alto


# poi per SibSp e ParCh li sommo per vedere quanta gente sta in una famiglia

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 per includere il passeggero stesso
df=df.drop(columns=['SibSp', 'Parch'])

## Plot tasso di sopravvivenza per dimensione della famiglia
#family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
#
#plt.figure(figsize=(8, 5))
#sns.barplot(x=family_survival.index, y=family_survival.values, color="steelblue")
#plt.xlabel('Family Size')
#plt.ylabel('Survival Rate (%)')
#plt.title('Survival Rate by Family Size')
#plt.tight_layout()
#plt.show()
#
#
## Plot quantità di passeggeri per dimensione della famiglia e classe
#plt.figure(figsize=(10, 6))
#sns.countplot(data=df, x='FamilySize', hue='Pclass', palette='Set2')
#plt.xlabel('Family Size')
#plt.ylabel('Count')
#plt.title('Count of Family Size per Pclass')
#plt.legend(title='Pclass')
#plt.tight_layout()
#plt.show()
#
## Plot tasso di sopravvivenza per dimensione della famiglia e classe
#family_pclass_survival = df.groupby(['FamilySize', 'Pclass'])['Survived'].mean().unstack() * 100
#
#plt.figure(figsize=(10, 6))
#family_pclass_survival.plot(kind='bar', figsize=(10, 6))
#plt.xlabel('Family Size')
#plt.ylabel('Survival Rate (%)')
#plt.title('Survival Rate by Family Size and Pclass')
#plt.legend(title='Pclass')
#plt.tight_layout()
#plt.show()

print(df.info())

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

print(df.head(1))

#df=df.dropna()
#
##random forest
#from sklearn.model_selection import train_test_split
#X = df.drop(columns=['Survived'])
#y = df['Survived']
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=100, random_state=42)
#rf.fit(X_train, y_train)
#from sklearn.metrics import classification_report, confusion_matrix
#
#print("train set")
#print(confusion_matrix(y_train, rf.predict(X_train)))
#print(classification_report(y_train, rf.predict(X_train)))
#print("Test set")
#print(confusion_matrix(y_test, rf.predict(X_test)))
#print(classification_report(y_test, rf.predict(X_test)))


#fatto tutto il preprocessing, salvo i dataset
#mo non ho voglia di farlo ma il training verrà fatto in un altro file

df_test = df[df['Survived'].isnull()].copy()
df_test = df_test.drop(columns=['Survived'])
df_train = df[df['Survived'].notnull()].copy()

df_train.to_csv('dataset/train_processed.csv', index=False)
df_test.to_csv('dataset/test_processed.csv', index=False)