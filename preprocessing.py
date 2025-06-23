import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

df = pd.concat([train_df, test_df], ignore_index=True)

print(df.info())

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

# gestione età mancanti, utilizzo della mediana per sesso e classe
median_ages = df.groupby(['Sex', 'Pclass'])['Age'].median()
print(median_ages)
df['Age'] = df.apply(
    lambda row: median_ages[row['Sex'], row['Pclass']] if pd.isna(row['Age']) else row['Age'],
    axis=1
)

# gestione "embarked" mancanti, mancano solo per due passeggeri, cercandole su internet
#Martha Evelyn Stone (830) si è imbarcata a Southampton assieme la domestica Amelie Icard (62)
df.loc[df['PassengerId'] == 830, 'Embarked'] = 'S'
df.loc[df['PassengerId'] == 62, 'Embarked'] = 'S'

#gestione "Fare" mancante, una sola entry è assente
#Thomas Storey (1044) in 3° classe, imbarcato senza parenti
#prendo il valore mediano della tariffa per 3° classe di tutti i passeggeri con SibSp=0 e Parch=0
df.loc[df['PassengerId'] == 1044, 'Fare'] = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

print(df.info())